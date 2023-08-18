import configparser
import math

from model import *
from conf import *
import dataset as Dataset
from torchmetrics import SpearmanCorrCoef
import pandas as pd
import pickle
import os
import datetime as dt
from clearml import Task
import datetime


def load_from_pickle(path):
    print(f"Loading : {path}")
    file = open(path, 'rb')
    obj = pickle.load(file)
    return obj


def get_clearml_task(name, lr, dt_string):
    now = dt.datetime.now()
    Task.set_credentials(api_host="https://api.community.clear.ml", key="3QZEM1KM8MBNZ115MFY6",
                         secret="EtmpxtZKXBe6LquSSQ3LobZhEarr8RQlZX9nEqqQbdYXFzvhRv")

    task = Task.init(project_name="pytorch_ch3", task_name=name)
    # task.set_credentials(api_host='https://app.community.clear.ml', key='0A4SJD10MUTBJ8KKPGAH')
    task.upload_artifact('run.py', artifact_object='run.py')
    task.upload_artifact('conf.py', artifact_object='conf.py')
    task.upload_artifact('dataset.py', artifact_object='dataset.py')
    
    task.add_tags("train data")
    task.add_tags(dt_string)
    task.set_comment(f"data path {Conf.PATH}")
    task.set_parameters({'learning_rate': lr, 'batch_size': Conf.batch_size})
    return task


def store_test_data_in_pickle(test, folder, task):
    print("Saving test data to pickle")
    pickle.dump(test, open(folder + f'/test.pkl', 'wb'))
    # task.upload_artifact(Conf.PATH + 'test.pickle', artifact_object='test.pickle')


def log_metrics_in_clearml(task, ii, batch_train_loss, batch_val_loss, batch_train_corr, batch_val_corr):
    # Log metrics to ClearML by batches
    # Log metrics to ClearML by batches
    task.get_logger().report_scalar(title="Loss", series="batch_train_loss", iteration=ii,
                                    value=batch_train_loss)
    task.get_logger().report_scalar(title="Loss", series="batch_val_loss", iteration=ii,
                                    value=batch_val_loss)
    task.get_logger().report_scalar(title="Corr", series="batch_train_corr", iteration=ii,
                                    value=batch_train_corr)
    task.get_logger().report_scalar(title="Corr", series="batch_val_corr", iteration=ii,
                                    value=batch_val_corr)


def read_data(task, folder):
    train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind = Dataset.read_data_sets(
        filename_sequence=Conf.filename_sequence,
        filename_expression=Conf.filename_expression,
        filename_labels=Conf.filename_labels,
        filename_dist=Conf.filename_dist,
        train_portion_subjects=Conf.train_portion_probes,
        train_portion_probes=Conf.train_portion_probes, validation_portion_subjects=Conf.validation_portion_subjects,
        validation_portion_probes=Conf.validation_portion_probes, directory=Conf.PATH, is_prediction=False,
        load_model_ID=0)
    
    print(f"train size :{train.num_examples}")
    print(f"val size :{validation.num_examples}")
    print(f"test size :{test.num_examples}")
    
    store_test_data_in_pickle(test, folder, task)
    return train, test, validation


def compute_early_stopping(iterations_without_improvement, ii, avg_val_cor, prev_val_avg_corr):
    early_stopping_flag = False
    if avg_val_cor - prev_val_avg_corr <= Conf.min_delta and ii > 1000:
        iterations_without_improvement += 1
        if iterations_without_improvement >= Conf.max_iterations_without_improvement:
            print("avg_val_cor", avg_val_cor)
            print("prev_val_avg_corr", prev_val_avg_corr)
            print(
                f"No improvement in the val correlation for {iterations_without_improvement} iterations. Stopping training.")
            early_stopping_flag = True
    else:
        iterations_without_improvement = 0
    return early_stopping_flag, iterations_without_improvement


def run_test_eval(model, test, batch_size, task, fast_test = False):
    test_losses = [];
    test_corrs = []
    test_size = test.num_examples
    print(f"test size :{test.num_examples}")
    if fast_test and test_size > 2000:
        test_size = 1000
    n_batches = math.ceil(test_size / batch_size)
        
    task.get_logger().report_single_value(name="test size", value=test_size)

    print(f"Runing test evaluation... number of test batches =  {n_batches}")
    model.eval()
    with torch.no_grad():
        for ii in range(n_batches):
            test_seq, test_exp, test_dist, test_labels = get_next_batch(test, batch_size, testing=True)
            task.add_tags("test size: {}".format(len(test_labels)))
            test_loss, test_corr = eval(test_seq, test_exp, test_dist, test_labels, model)
            test_losses.append(test_loss)
            test_corrs.append(test_corr)
            print(f"{ii}")
            print("Test| Loss: {:.3f} Corr: {:.3f}".format(test_loss, test_corr))

        avg_test_loss = np.mean(test_losses)
        avg_test_corr = np.mean(test_corrs)
        print("Test overall | Loss: {:.3f} Corr: {:.3f}".format(avg_test_loss, avg_test_corr))

    # logging test results to clearml
    task.get_logger().report_single_value(name="test_loss", value=avg_test_loss)
    task.get_logger().report_single_value(name="test_corr", value=avg_test_corr)
    return avg_test_loss, avg_test_corr


def run_validation(model, loss_fn, validation, epoch_val_losses, epoch_val_corrs):
    # Compute validation metrics
    with torch.no_grad():
        val_seq_batch, val_exp_batch, val_dist_batch, val_labels_batch = get_next_batch(validation, Conf.batch_size)

        val_predictions = model(val_seq_batch, val_dist_batch, val_exp_batch)

        batch_val_loss = loss_fn(val_predictions, val_labels_batch)
        epoch_val_losses.append(batch_val_loss.cpu())  # todo: why cpu?

        batch_val_corr = SpearmanCorrCoef()(val_predictions.squeeze(), val_labels_batch.squeeze())
        epoch_val_corrs.append(batch_val_corr.cpu())  # todo: why cpu?
        return batch_val_loss, batch_val_corr, epoch_val_losses, epoch_val_corrs


def save_model_to_pickle(model, folder_name, name, val_cor, epoch=-1, ii=-1):
    if ii > 0:
        file_name = f'{name}_{epoch}_{ii}'
    else:
        file_name = f'{name}_final'

    print(f"iteration : {ii} Saving model to pickle as {folder_name}{file_name}")
    path = f"{folder_name}/{file_name}"
    pickle.dump(model, open(path + ".pkl", 'wb'))
    torch.save(model.state_dict(), path + ".pt")


def run_training():
    # learning_rates = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    learning_rates = [0.0001]
    factor = 0.1

    for lr in learning_rates:
        early_stopping_flag = False
        iterations_without_improvement = 0
        prev_val_avg_corr = float('-inf')

        current_datetime_str = dt.datetime.now().strftime('%m_%d_%H_%M_%S')
        folder_name = f"{Conf.PATH}/{current_datetime_str}"
        os.mkdir(folder_name)
        print(f"Created train folder : {folder_name}")

        task_name = "ch3 full model training"
        task = get_clearml_task(task_name, lr, current_datetime_str)

        print(f"Starting train with learning rate = {lr}")

        train, validation, test = read_data(task, folder_name)

        # Set up your model, loss function, and optimizer
        if Conf.load_model:
            multi_model = load_from_pickle(Conf.model_path)
        else:
            multi_model = MultiModel()
            if torch.cuda.is_available():
                multi_model = multi_model.cuda()
            multi_model = multi_model.float()

        optimizer = optim.Adam(multi_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=10)
        loss_fn = nn.L1Loss()

        train_losses = [];
        val_losses = [];
        train_corrs = [];
        val_corrs = [];

        # Training loop
        for epoch in range(Conf.epochs):
            if early_stopping_flag:
                break
            print(f"Epoch {epoch}")
            epoch_val_losses = [];
            epoch_train_losses = [];
            epoch_val_corrs = [];
            epoch_train_corrs = []

            print(f"Train data number of batches : {train.num_examples // Conf.batch_size}")
            for ii in range(train.num_examples // Conf.batch_size):
                if early_stopping_flag:
                    break
                multi_model.train()
                train_seq_batch, train_exp_batch, train_dist_batch, train_labels_batch = get_next_batch(train,
                                                                                                        Conf.batch_size)
                optimizer.zero_grad()
                # Forward pass
                train_predictions = multi_model(train_seq_batch.float(), train_dist_batch.float(),
                                                train_exp_batch.float())

                # Compute the loss & correlation
                batch_train_loss = loss_fn(train_predictions, train_labels_batch)
                epoch_train_losses.append(batch_train_loss.cpu().detach().numpy())

                batch_train_corr = SpearmanCorrCoef()(train_predictions.squeeze(), train_labels_batch.squeeze())
                epoch_train_corrs.append(batch_train_corr.cpu())

                # Backward pass and optimization
                batch_train_loss.backward()
                optimizer.step()
                # scheduler.step(epoch_val_avg_loss)

                # Set the model to evaluation mode
                multi_model.eval()  # adi: this was with a comment out

                batch_val_loss, batch_val_corr, epoch_val_losses, epoch_val_corrs = \
                    run_validation(multi_model, loss_fn, validation, epoch_val_losses, epoch_val_corrs)

                if ii % 1 == 0:
                    print(f"epoch : {epoch} batch num :{ii}")
                    print("Train      | Loss: {:.3f} Corr: {:.3f}".format(batch_train_loss, batch_train_corr))
                    print("Validation | Loss: {:.3f} Corr: {:.3f}".format(batch_val_loss, batch_val_corr))

                    # Log metrics to ClearML by batches
                    log_metrics_in_clearml(task, ii, batch_train_loss, batch_val_loss, batch_train_corr, batch_val_corr)

                if ii % 20 == 0 and batch_val_corr > Conf.save_model_th:
                    save_model_to_pickle(multi_model, folder_name, task_name, val_cor=batch_val_corr, epoch=epoch,
                                         ii=ii)

                avg_val_cor = np.mean(epoch_val_corrs[-10:])

                early_stopping_flag, iterations_without_improvement = compute_early_stopping(
                    iterations_without_improvement, ii,
                    avg_val_cor, prev_val_avg_corr)

                prev_val_avg_corr = avg_val_cor

            # scheduler.step(np.mean(epoch_val_losses))
            epoch_train_avg_loss = np.mean(epoch_train_losses)
            epoch_train_avg_corrs = np.mean(epoch_train_corrs)
            train_losses.append(epoch_train_avg_loss)
            train_corrs.append(epoch_train_avg_corrs)

            epoch_val_avg_loss = np.mean(epoch_val_losses)
            epoch_val_avg_corrs = np.mean(epoch_val_corrs)
            val_losses.append(epoch_val_avg_loss)
            val_corrs.append(epoch_val_avg_corrs)

            task.get_logger().report_single_value(name="train size", value=train.current_position)

        save_model_to_pickle(multi_model, folder_name, task_name, val_cor=np.mean(val_corrs))
        print("Finish training")
        run_test_eval(multi_model, test, Conf.batch_size, task)
        task.close()


def main():
    print(f"data path : {Conf.PATH}")
    run_training()


if __name__ == "__main__":
    main()
