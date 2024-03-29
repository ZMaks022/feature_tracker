# Pyramidal Implementation of the Lucas Kanade Feature Tracker
___
![](examples/output.gif)

### Installation

To run this project on MacOS, follow these steps.

#### Installing OpenCV
To install OpenCV using Homebrew, open a terminal and enter the following command:

```bash
brew install opencv
```

#### Preparing directories
Create directories for input and output data:

```bash
mkdir inputs
mkdir outputs
```

#### Extracting frames from GIF
Use ffmpeg to extract frames from a GIF file. Before running this command, make sure that the 4.gif file is in a folder one level above your current working directory.

```bash
cd inputs
ffmpeg -i ../4.gif frame_%03d.png
```

### Generate GIF
```bash
ffmpeg -framerate 24 -start_number 5 -i outputs/frame_%03d.png -vf "fps=24,scale=320:-1:flags=lanczos,palettegen" palette.png
ffmpeg -framerate 24 -start_number 5 -i outputs/frame_%03d.png -i palette.png -filter_complex "fps=24,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse" output.gif
```

### Examples
#### Horizontal movement
frame 015 before features tracking

![](examples/frame_015.png)

frame 025 after features tracking, we can see how every point moved to new position.

![](examples/frame_025.png)

#### Vertical movement
frame 55 before features tracking

![](examples/frame_055.png)

frame 65 after features tracking, we can see how every point moved to new position.

![](examples/frame_065.png)
