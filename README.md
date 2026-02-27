# SketchVision
[USENIX NSDI '26](https://www.usenix.org/conference/nsdi26/presentation/mirnajafizadeh)

## Introduction
**SketchVision** is a vision-inspired detection framework designed to defend Content Delivery Networks (CDNs) against slow-and-low attacks. It overcomes limitations in edge defense by efficiently monitoring flow behavior and detecting attacks under resource-constrained settings. Using a sketch that encodes packet-level temporal patterns into compact images, a diffusion model for sketch denoising, and a generative inference pipeline, SketchVision achieves early and robust detection across multiple attack types.

<!-- <div align="center" style="padding: 12px; border: 1px solid #ddd; border-radius: 12px;">
  <img src="./SketchVision.png" width="250" height="200"/>
</div> -->

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
   invoke run-test
===============================
```

This confirms that the Docker environment is ready. You can now use invoke commands to build, compile, and run tests.
Additional details can be found [here](https://github.com/NIDS-LAB/SketchVision/tree/main/SketchVision).



