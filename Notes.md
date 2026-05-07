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

weight_theta = scale_theta / (2 * dx) =  211.6588 / (2 * 0.045.. rad) = 2609

# ==== for r ==== 

weight_r = scale_r / (2 * dx ) = 5.02 / (2 * dx 0.2693 m )= 9.33


____ 
## Correlation area calculation

mean value (r, theta) between following steps:

dr_mean = 0.2693 m
dtheta_mean = 0.045 rad

dr_max = 0.2693 m
dtheta_max = 0.045 rad

if means, after max step, object can change posiiton on image:

v = (2 * 0.2693 m * 5.02 pix/m, 2 * 0.045 rad * 211.6588 pix/rad) = (2.70, 19.04)

on feature map, which is downsized 4 times it became: 
v_fmap = v / 4 =(0.675, 4.76).

So, it is necessery to search at least area (-5, 5) from patch center on fmap. That will cover (-20, 20) pix on original image. 

I set search_size = 7, it covers (-3, 3) on fmap. 

2-nd lvl pyramid. 
I passed fmap through avg_pool2d. It downsampled 2 times fmap. I calculate correlation on search size, 
Then correlation of my patch is calculate with each search_size * search_size area. 
It gives us again, search size (-3, 3) on downsample fmap -> (-6, 6) on orginal fmap -> 2nd lvl fmap can see the biggest considered movement. 
is it enough? 




