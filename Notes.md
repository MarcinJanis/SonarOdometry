#TODO:



## How many pixels corresponds to 1 meter in real world?

1. For **x axis** (theta) 

Sonar horizntal fov = 2.2678 rad = 130 deg
x-axis resolution of fls img: 480 

**scale_theta = 480/130 = 211.6588 pix/rad (or 0.0047 rad per pix)**


1. For **y axis** (r) 

Sonar range = 0.5 - 100 m, range = 99.5 m
y-axis resolution of fls img: 500 

**scale_r = 500/99.5 = 5.02 pix/m (or 0.325 m per pix)**

## How to choose weight for dumper in Bundle adustment 

When step goes to its max tolerable value (we assume that is mean value of movemnt between next steps in dataset), dumper "power" shall be equal to reprojection loss "power"
When using L1, when error is bigger than Betha = 1.0, gradient is equal -1.0, else it use L2. 
We consider only L1 case, L1_grad = 1.0

Grad from dumper is weight*(dx^2)' = 2*dx*weight (dx is related to initial pose, dx is this value scaled to fls pixel values) 

In consequence: 

2*dx*weight = 1.0 * scale [pix/unit]
weight = 0.5 / dx = scale [pix/unit] / 2*dx 

# ==== for theta ==== 

---------------------------------------------------------dx_pix = dx*scale = 0.04056 rad * 211.6588 pix/rad = 8.5848 pix for mean rotation movement 

weight_theta = scale_theta / (2 * dx) = 2609

# ==== for r ==== 

---------------------------------------------------------dx_pix = dx*scale = 0.2693 m * 5.02 pix/m = 1.35188 pix for mean rotation movement 
weight_r = scale_r / (2 * 0.2693 m ) = 9.33




