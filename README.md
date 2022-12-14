# CT_segmentation_test_task

## Описание задания

Нужно написать бейзлайн для обучения сегментационной нейросети для обнаружения плеврального выпота. 

Обучить можно любую сегментационную нейросеть, можно использовать 2D или 3D архитектуру.

Код должен быть разделен следующим образом:

- файл с архитектурой модели
- файл с препроцессингом данных для обучения
- файл с датасетом
- файл с функцией подсчета [DICE Coef](https://radiopaedia.org/articles/dice-similarity-coefficient#:~:text=The%20Dice%20similarity%20coefficient%2C%20also,between%20two%20sets%20of%20data.)
- основной файл с самим циклом обучения. В коде обучения во время валидации автоматически должна  выбираться лучшая эпоха и сохраняться веса модели в папку `output`. Также после цикла обучения в эту папку должна сохраняться картинка с изменениями коэффициента DICE с каждой эпохой. (по оси Y - коэффициент DICE, по оси X - номер эпохи). Вместо выходной картинки с графиком можно использовать любые трекеры при желании (tensorboard etc.)

## Описание решения

В качестве модели была выбрана monai.networks.nets.unet. Датасет из 10 снимков разделен на 8 (train) и 2 (valid). 

Препроцессинг КТ снимков: нормализация, сжатие в 2 раза по осям 2, 3 (до размера D\*256\*256), удаление фона.

Для обучения использованы дополнительные аугментации: RandCropByPosNegLabeld с roi=(48, 96, 96), повороты по осям.

Обучение модели происходит на кусках размером (48, 96, 96). Инференс происходит с помощью скользящего окна данного размера с перекрытием в 75%.

В качестве лосс функции используется DiceLoss и BCELoss с pos_weight=3. 

В качестве основы решения используется Pytorch Lightning, для трекинга WandB.

Ссылка на результаты эксперимента: https://wandb.ai/mtyutyulnikov/lightning_logs/runs/24h25hvd?workspace=user-mtyutyulnikov

Максимальная метрика Dice на валидационной подвыборке составила 0.6277.

<img width="900" alt="wandb_screenshot" src="https://user-images.githubusercontent.com/64151232/205306072-2acae7cd-ed42-4fce-ac80-6e4d1a24a0f0.png">
