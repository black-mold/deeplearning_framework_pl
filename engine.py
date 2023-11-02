import lightning.pytorch as pl
import torch




class FeatureExtractor(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler):
        super().__init__()
        # ⚡ model
        self.model = model
        print(self.model)

        # ⚡ loss 
        self.loss_function = loss_function

        # ⚡ optimizer
        self.optimizer = optimizer

        # ⚡ scheduler
        self.scheduler = scheduler # **kwargs: **config['scheduler_config']

        # save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        #⚡⚡⚡ debugging - print input output layer ⚡⚡⚡
        self.example_input_array = torch.Tensor(64, 1, 28, 28)

        # for validation & test
        self.training_step_outputs = [] # not used, but I want to keep it for future implementation
        self.validation_step_outputs = []


    # ===============================================================
    # ⚡⚡ Train
    # ===============================================================


    def training_step(self, batch, batch_idx):
        x, y = batch
        # preprocess

        # inference
        y_hat = self.model(x)

        # post processing

        # calculate loss
        loss = self.loss_function(y_hat, y)

        self.training_step_outputs.append(loss)
        # Logging to TensorBoard
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True)        

        return loss

    def on_train_epoch_end(self):
        self.training_step_outputs.clear() # free memory

    # ===============================================================
    # ⚡⚡ Validation
    # ===============================================================
    

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss,  on_epoch= True, prog_bar=True, logger=True)

        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        size = x.shape[0]

        validation_step_output = {'correct': correct, 'size': size}

        self.validation_step_outputs.append(validation_step_output)
        return validation_step_output

    def on_validation_epoch_end(self):

        correct_score = sum([dic['correct'] for dic in self.validation_step_outputs])
        total_size = sum([dic['size'] for dic in self.validation_step_outputs])
        acc = correct_score/total_size

        self.log("val_ACC", acc * 100, on_epoch = True, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear() # free memory

    # ===============================================================
    # ⚡⚡ test
    # ===============================================================
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self,  test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)
    


    def forward(self, x):
        y_hat = self.model(x)
        return y_hat


    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "name": "lr_log",
            },
        }