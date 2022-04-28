# PoryDrive
A 3D driving game.

[![Screenshot of the PoryDrive game](https://raw.githubusercontent.com/mrbid/porydrive/main/screenshot.png)](https://youtu.be/qutsFPF7tH8 "PoryDrive Game Video")

I have always liked racing games, ever since the original V-Rally I was fascinated by the bouncy "realistic" physics of the vehicles, in 2006 I would make my first attempt at a 3D car game in C++ using the [Ogre3D rendering engine](https://www.ogre3d.org/) in the game [TeapotHunt](https://github.com/traxpuzzle/Misc-Games/raw/main/TeapotHunt.exe); basic, and with no acceleration or deceleration because at that point I did not really understand floating-point numbers. By 2009 my Math had improved a lot thanks to the help of my university teachers, most specifically it was [Tyrone Davison](https://research.tees.ac.uk/en/persons/tyrone-davison) who taught me or well helped me realise the importance of floating-point math. In my final year of university, I used Bullet Physics and DirectX 9 with C++ to make another racing game called [Burning Rubber](https://github.com/mrbid/DX9-Racing-Game) with some friends for the Team Working module. It has been a decade since then and I am overjoyed that I have finally revisited this passion of mine having created a basic car physics simulation from scratch with self-guessed knowledge of car physics.

## How to play
Drive around and "collect" Porygon, each time you collect a Porygon a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help you find them. Upon right clicking the mouse you will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.

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
steeringtransfer 0.019
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

The Machine Learning or "neural" agents use a modified version of [TFCNNv1](https://github.com/mrbid/TFCNNv1/blob/main/TFCNNv1.h) which optimises for a tanh output. The dataset was captured from ~6.4 hours of the Auto Drive algorithm playing the game _(not supplied due to file size, please train your own, read the [`trainer.c`](trainer.c) header)_. Two agents are trained from this dataset, the `steering agent` and `gasing agent` respectively. One controlling steering and the other controlling speed. The file [`globaldef.h`](globaldef.h) holds the variables that configure the size _(HIDDEN_SIZE)_ and complexity _(HIDDEN_LAYERS)_ of the neural network. If these definitions are changed the neural network will need to be retrained by executing [`train.sh`](train.sh). Other hyperparameters can be configured in the [`trainer.c`](trainer.c) file itself after the `createNetwork()` function.

There is one disadvantage to these two systems; they break the laws of acceleration and turn speed. These algorithms can set arbitrary wheel turn angles and speeds with disregard to acceleration limits or wheel turn speed limits. A post-process could be added to enforce that transitions on these variables stay within such limits.

The input training data is _(4 byte float32 per parameter)_;
- car normal dir x
- car normal dir y
- (car_pos - porygon_pos) normal dir x
- (car_pos - porygon_pos) normal dir y
- angle between directions
- distance between car and porygon

Training data targets _(one target per trained network)_
- car steering angle
- car speed

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

