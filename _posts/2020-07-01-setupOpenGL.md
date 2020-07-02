---
layout: post
title:  "Setting up Modern OpenGL on Linux"
date:   2020-07-01 
tags: [computer graphics, rasterization, OpenGL]
feature: assets/img/setupOpenGL/initial_img.png
---

In this article I'll show how to set up Modern OpenGL on Linux and how to run a simple image.

---

## Dependencies
The compilation of the project requires that OpenGL headers and GLUT (The OpenGL Toolkit)
installed.

### Dependencies Installation on Linux

**Debian, Ubuntu**
```
$ sudo apt-get install build-essential libgl1-mesa-dev
$ sudo apt-get install libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev
```

**Fedora**
```
$ sudo dnf install make gcc-c++
$ sudo dnf install glew-devel SDL2-devel SDL2_image-devel glm-devel freetype-devel
```
If GLM is not available in your distribution repository, you have the option to install it manually. Make sure the headers end in the */usr/include/glm directory*. Since it's a headers-only library, you do not need to compile a .so library - just copy the code there.


### Check your OpenGL installation

```
$ glxinfo | grep OpenGL
```

---

## Compilation Instructions

**Compilation**
```
$ make
```

**Cleaning**
```
$ make clean
```

**Run**
```
$ ./moderngl_hello_world
```

---

## Example

<p align="center">
<strong>The result should be something like this:</strong>
</p>

<div align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/Ug2uIIbkVcU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

---

## Repository

Check it out on [Github](https://github.com/jpvt/Computer_Graphics/tree/master/Assignment%20%232).