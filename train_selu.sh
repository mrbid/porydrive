rm train_selu
gcc selu_trainer.c -lm -Ofast -o train_selu
sudo ./train_selu 0
sudo ./train_selu 1
