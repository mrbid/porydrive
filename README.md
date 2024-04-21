## **Latest GLFW/WebGL Fork:** https://github.com/mrbid/PoryDrive-2.0
The above link has the latest fun version of the game to play, the one in this repository is the first release which is a bit of a mess.

# PoryDrive
A 3D driving game with neural networks.

[![Screenshot of the PoryDrive game](https://raw.githubusercontent.com/mrbid/porydrive/main/screenshot.png)](https://www.youtube.com/watch?v=EoPIzxVX-9E "PoryDrive Game Video")

---

* **Play Online:** https://mrbid.github.io/porydrive
* **Flathub:** https://flathub.org/apps/com.voxdsp.PoryDrive
* **Windows:** https://notabug.org/attachments/d49923d5-230c-408a-9440-693ca8462bb6
* **Source Code:** https://notabug.org/Vandarin/PoryDrive

---

## [CLICK HERE FOR THE LINUX MACHINE LEARNING FORK](https://github.com/PoryDrive/PoryDriveFNN)
### This game is capped to 100-144 FPS, if your machine cannot keep up it will just simulate the game a little slower. The latest linux fork has much better timing code. [selu_main.c](selu_main.c) and [keras_main.c](keras_main.c) use an adaptive framerate but the game simulation gets worse below 100 fps and unplayable under 30 fps.

---

I have always liked racing games, ever since the original V-Rally I was fascinated by the bouncy "realistic" physics of the vehicles, in 2006 I would make my first attempt at a 3D car game in C++ using the [Ogre3D rendering engine](https://www.ogre3d.org/) in the game [TeapotHunt](https://github.com/traxpuzzle/Misc-Games/raw/main/TeapotHunt.exe); basic, and with no acceleration or deceleration because at that point I did not really understand floating-point numbers. By 2009 my Math had improved a lot thanks to the help of my university teachers, most specifically it was [Tyrone Davison](https://research.tees.ac.uk/en/persons/tyrone-davison) who taught me or well helped me realise the importance of floating-point math. In my final year of university, I used Bullet Physics and DirectX 9 with C++ to make another racing game called [Burning Rubber](https://github.com/mrbid/DX9-Racing-Game) with some friends for the Team Working module. It has been a decade since then and I am overjoyed that I have finally revisited this passion of mine having created a basic car physics simulation from scratch with self-guessed knowledge of car physics.

## How to play
Drive around and "collect" Porygon, each time a Porygon is collected a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help find them. Upon right clicking the mouse the view will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.

## Keyboard
 - `ESCAPE` = Focus/Unfocus Mouse Look
 - `N` = New Game
 - `W` = Drive Forward
 - `A` = Turn Left
 - `S` = Drive Backward
 - `D` = Turn Right
 - `Space` = Breaks
 - `1-5` = Car Physics config selection _(5 loads from file)_

## Keyboard Dev
 - `F` = FPS to console
 - `P` = Player stats to console
 - `O` = Toggle auto drive
 - `I` = Toggle neural drive
 - `L` = Toggle dataset logging

## Mouse
 - `Mouse Button4` = Zoom Snap Close/Ariel
 - `Mouse Click Right` = Zoom Snap Close/Ariel
 - `Middle Scroll` = Zoom in/out

## Configure Car Physics
It is possible to tweak the car physics by creating a `config.txt` file in the exec/working directory of the game, here is an example of such config file with the default car physics variables.
```
maxspeed 0.0095
acceleration 0.0025
inertia 0.00015
drag 0.00038
steeringspeed 1.2
steerinertia 233
minsteer 0.32
maxsteer 0.55
steeringtransfer 0.023
steeringtransferinertia 280
```
- `maxspeed` - top travel speed of car.
- `acceleration` - increase of speed with respect to time.
- `inertia` - minimum speed before car will move from a stationary state.
- `drag` - loss in speed with respect to time.
- `steeringspeed` - how fast the wheels turn.
- `steerinertia` - how much of the max steering angle is lost as the car increases in speed _(crude steering loss)_.
- `minsteer` - minimum steering angle as scalar _(1 = 180 degree)_ attainable after steering loss caused by `steeringintertia`.
- `maxsteer` - maximum steering angle as scalar _(1 = 180 degree)_ attainable at minimal speeds.
- `steeringtransfer` - how much the wheel rotation angle translates into rotation of the body the wheels are connected to _(the car)_.
- `steeringtransferinertia` - how much the `steeringtransfer` reduces as the car speed increases, this is related to `steerinertia` to give the crude effect of traction loss of the front tires as speed increases and the inability to force the wheels into a wider angle at higher speeds.

## Auto Drive & Neural Drive
Auto Drive is based on a simple concept similar to that of the [Wall follower maze-solving algorithm](https://en.wikipedia.org/wiki/Maze-solving_algorithm#Wall_follower) the principle relies on the fact that if you always turn in one direction and aim to reduce the angle of direction between the agent and the target you will eventually hit the target. On top of this, there is a rule to switch the turn direction if the distance from the target increases by more than a threshold scaled by a scalar based on the distance from the target. It gives a pretty good effect and with competitive results to human players.

The Machine Learning or "neural" agents use a modified version of [TFCNNv1](https://github.com/mrbid/TFCNNv1/blob/main/TFCNNv1.h) which optimises for a tanh output. The dataset was captured from ~6.4 hours of the Auto Drive algorithm playing the game with the ScarletFast car physics config _(dataset not supplied due to file size, please train your own, read the [`trainer.c`](trainer.c) header)_. Two agents are trained from this dataset, the `steering agent` and `gasing agent` respectively. One controlling steering and the other controlling speed. The file [`globaldef.h`](globaldef.h) holds the variables that configure the size _(HIDDEN_SIZE)_ and complexity _(HIDDEN_LAYERS)_ of the neural network. If these definitions are changed the neural network will need to be retrained by executing [`train.sh`](train.sh). Other hyperparameters can be configured in the [`trainer.c`](trainer.c) file itself after the `createNetwork()` function.

There is one disadvantage to these two systems; they break the laws of acceleration and turn speed. These algorithms can set arbitrary wheel turn angles and speeds with disregard to acceleration limits or wheel turn speed limits. A post-process could be added to enforce that transitions on these variables stay within such limits.

### The input training data _(4-byte float32 per parameter)_
- car normal dir x
- car normal dir y
- (car_pos - porygon_pos) normal dir x
- (car_pos - porygon_pos) normal dir y
- angle between both normal dir's _(Dot product)_
- Euclidean distance between car and porygon

### Training data targets _(one target per trained network)_
- car steering angle
- car speed

### Training Results
- `HIDDEN_LAYERS` 1
- `HIDDEN_SIZE` 64
- `OPTIM_ADAGRAD`
- `DROPOUT` 0.3
- `BATCHES` 32
- `ACTIVATOR` tanh
- `WEIGHT_INIT_UNIFORM_LECUN`

Good results for a small 73kb network. Sometimes locates porygon quickly. Weights provided [steernet](steeragent_weights.dat) and [gasnet](gasagent_weights.dat).

_Notes: Increasing the hidden layers or dropout seems to have a negative effect on the trained results._

In hindsight I think the Steering Agent may have had better results without being trained using Euclidean distance as an input. Just the normal directions may have been enough information because it would probably have been easier to optimise for.

### Next Steps
- ~~Try a SELU network [TFCNNv2.1](https://github.com/mrbid/TFCNNv2/tree/main/TFCNNV2.1) to see if this has improvements with more hidden layer depth.~~ _(SELU support has been added, needs some tweaking still as I am yet to get good trained results, WIP, [selu_trainer.c](selu_trainer.c) & [selu_main.c](selu_main.c), honestly I don't think it's worth the time investment when Tensorflow Keras will yeild better results with less time invested.)_
- ~~Try some networks in Tensorflow Keras, FNN with ADAM and a CNN.~~ _(the Keras version [keras_main.c](keras_main.c) & [train.py](train.py) trains one network with two tanh outputs, one for steering and one for speed.)_

## Training with Tensorflow Keras
Keras trains much better than my personal neural networks, partly because Keras supports the superior [ADAM optimiser](https://keras.io/api/optimizers/adam/).

To train the network in Keras first you need to split the `dataset.dat` file into x targets (input data) and y targets (input labels for training). This is done by compiling and running the [splitter.c](splitter.c) program which will output `dataset_x.dat` and `dataset_y.dat`. Then you need to run [train.py](train.py) which will output `porygon_model/keras_model`. Then to boot the PoryDrive game you need to run [pred.py](pred.py) which will execute a daemon that will take data from the PoryDrive game using a RAM file in `/dev/shm/porydrive_input.dat`, run it through the Keras model, and then return the result back to a second RAM file `/dev/shm/porydrive_r.dat`. Once `pred.py` is running you can compile and execute the Keras version of PoryDrive by executing `compile_keras.sh`.

The hyperparameters for the Keras network can be configured near the header of [train.py](train.py).

It doesn't really seem to get any better than 1 hidden layer with 32 - 64 units.

It would be interesting to now train this Keras model on data collected from human input and not the simple "AutoDrive" algorithm.

I have forked this project to it's own organisation to keep track of it's development better, in the new repositories I upload full datasets and manage the directory structure in a slightly cleaner and concise manner. If this is something you are interested in, the latest updates to this project in respect to Machine Learning are developed over at this git; https://github.com/PoryDrive

If you are interested in just the game with no Auto-Drive or Neural Networks then please refer to [Release6](https://github.com/mrbid/porydrive/tree/Release6).

## Downloads

### Snapcraft
https://snapcraft.io/porydrive

### AppImage
https://github.com/mrbid/porydrive/raw/main/PoryDrive-x86_64.AppImage

### [x86_64] Linux Binary (Ubuntu 21.10)
https://github.com/mrbid/porydrive/raw/main/porydrive

### [ARM64] Linux Binary (Raspbian 10)
TBA<br>

### Windows Binary
https://github.com/mrbid/porydrive/raw/main/porydrive.exe<br>
https://github.com/mrbid/porydrive/raw/main/glfw3.dll

## Attributions
https://downloadfree3d.com/3d-models/vehicles/car/bmw-e34/<br>
https://www.cgtrader.com/free-3d-models/character/fantasy-character/porygon<br>
https://www.cgtrader.com/free-3d-models/science/laboratory/dna-molecule-3d-model<br>
https://pixels.com/featured/1-bmw-m5-e34-tuning-bmw-5-series-tunned-m5-hervey-dopson.html<br>
https://gamerant.com/pokemon-sword-shield-porygon-porygon2-porygonz-isle-of-armor-dlc/ [**[`src`]**](https://www.pokemon.com/uk/pokemon-tcg/pokemon-cards/sm-series/sm10/155/)<br>
http://www.forrestwalter.com/icons/<br>

