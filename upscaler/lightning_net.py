import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from loss import HSVPerceptualLoss

class LitUpscaler(L.LightningModule):
    def __init__(self, model, lr=1e-6):
        super().__init__()
        self.model = model
        self.lr = lr

    def set_loss(self, loss):
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print("LR inside configure:", self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ImageLoggerCallback(L.Callback):
    def __init__(self, val_samples, log_every_n_epochs=1):
        """
        val_samples: DataLoader или список (x, y), на которых хотим логировать
        """
        self.val_samples = val_samples
        self.log_every_n_epochs = log_every_n_epochs

        x, y = next(iter(val_samples))
        self.fixed_x = x
        self.fixed_y = y


    def hsv_to_rgb(self, hsv):
        """
        hsv: Tensor [B,3,H,W] или [3,H,W] в диапазоне [0,1].
        return: rgb в [0,1]
        """
        h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]  # [B,H,W]

        h = h * 6  # 0..6
        i = torch.floor(h).long()              # целая часть 0..5
        f = h - i                              # дробная часть

        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        # каждый сектор HSV
        i = i % 6

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        mask = (i == 0)
        r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]

        mask = (i == 1)
        r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]

        mask = (i == 2)
        r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]

        mask = (i == 3)
        r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]

        mask = (i == 4)
        r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]

        mask = (i == 5)
        r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

        return torch.stack([r, g, b], dim=1)


    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        x = self.fixed_x.to(pl_module.device)
        y = self.fixed_y.to(pl_module.device)

        with torch.no_grad():
            pred = pl_module(x)

        # берём первые 4 картинки из батча
        img_gt   = y[:4].cpu()
        img_pred = pred[:4].cpu()

        img_gt = self.hsv_to_rgb(img_gt)
        img_pred = self.hsv_to_rgb(img_pred)

        # объединяем по каналу C для отображения GT сверху, предсказание снизу
        imgs = torch.cat([img_gt, img_pred], dim=2)  # вертикально
        grid = make_grid(imgs, nrow=4)  # 4 картинки в ряд

        pl_module.logger.experiment.add_image("GT_vs_Pred", grid, global_step=epoch)
