# SketchVision

## Introduction
**SketchVision** is a vision-inspired detection framework designed to defend Content Delivery Networks (CDNs) against slow-and-low attacks. It overcomes limitations in edge defense by efficiently monitoring flow behavior and detecting attacks under resource-constrained settings. Using a sketch that encodes packet-level temporal patterns into compact images, a diffusion model for sketch denoising, and a generative inference pipeline, SketchVision achieves early and robust detection across multiple attack types.

---

## Quick Start

All dependencies are resolved inside the Docker container. As long as you have a GPU (CPU works but slower), you can run the framework without additional setup.

Launch the environment:
```bash
./run_sketchvision-env.sh

Once inside, you should see:
=======================================
 Welcome to the SketchVision Framework
=======================================
Project location: /SketchVision
You can run any script, e.g.:
   invoke -l
   invoke build
   invoke compile
   invoke run_test
===============================
```

This confirms that the Docker environment is ready. You can now use invoke commands to build, compile, and run tests.



