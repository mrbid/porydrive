rm train
gcc trainer.c -lm -Ofast -o train
xterm -e "cd /home/v/Desktop/porydrive/;echo v | sudo -S ./train 0" &
xterm -e "cd /home/v/Desktop/porydrive/;echo v | sudo -S ./train 1" &