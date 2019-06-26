import cv2
import math
import progressbar
from pointillism import *




img_path = "port1-1024.jpg"
stroke_scale = 2
stroke_width = 1
gradient_smoothing_radius = 1
original_image_blur = 51 # I think this has to be an odd number
grid_scale = 3
color_k = 20
strokes_overlay_alpha = 0.4
batch_size = 100

palette_size = 20
limit_image_size = 0

result_path = img_path.rsplit(".", -1)[0] + "_drawing.jpg"
grad_mag_path = img_path.rsplit(".", -1)[0] + "_grad-mag.jpg"
grad_dir_path = img_path.rsplit(".", -1)[0] + "_grad-dir.jpg"

img = cv2.imread(img_path)

if limit_image_size > 0:
    img = limit_size(img, limit_image_size)

if stroke_scale == 0:
    stroke_scale = int(math.ceil(max(img.shape) / 1000))
    print("Automatically chosen stroke scale: %d" % stroke_scale)
else:
    stroke_scale = stroke_scale

if gradient_smoothing_radius == 0:
    gradient_smoothing_radius = int(round(max(img.shape) / 50))
    print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
else:
    gradient_smoothing_radius = gradient_smoothing_radius

# convert the image to grayscale to compute the gradient
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Computing color palette...")
palette = ColorPalette.from_image(img, palette_size)

print("Extending color palette...")
palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

# display the color palette
cv2.imwrite("pallete.jpg", palette.to_image())

print("Computing gradient...")
gradient = VectorField.from_gradient(gray)


#cv2.imwrite("grad.jpg", gradient.to_image())

print("Smoothing gradient...")
gradient.smooth(gradient_smoothing_radius)

##cv2.imwrite("grad_smooth.jpg", gradient.to_image())


res2 = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255


#for x in range(img.shape[0]):
#    for y in range(img.shape[1]):
#        color = (100,100,100)
#        angle = math.degrees(gradient.direction(y, x)) + 90
#        length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
#        cv2.line(res2, (x, y), (int(x+length*math.cos(angle)), int(y+length*math.sin(angle))), color)

cv2.imwrite(grad_mag_path, gradient.get_magnitude_image())

print("Drawing image...")
# create a "cartonized" version of the image to use as a base for the painting
res = cv2.medianBlur(img, original_image_blur)
#res = np.ones((img.shape[0], img.shape[1]), np.uint8) * 255


# define a randomized grid of locations for the brush strokes
grid = randomized_grid(img.shape[0], img.shape[1], scale=grid_scale)



bar = progressbar.ProgressBar()
for h in bar(range(0, len(grid), batch_size)):
    # get the pixel colors at each point of the grid
    pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
    # precompute the probabilities for each color in the palette
    # lower values of k means more randomnes
    color_probabilities = compute_color_probabilities(pixels, palette, k=color_k)

    for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
        color = color_select(color_probabilities[i], palette)
        angle = math.degrees(gradient.direction(y, x)) + 90
        #angle2 = math.degrees(gradient.direction(y, x)) + 90
        length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
        #length2 = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
        
        overlay = res.copy()
        # draw the brush stroke
        cv2.ellipse(overlay, (x, y), (length, 1), angle, 0, 360, color, -1, cv2.LINE_AA)
        res = cv2.addWeighted(overlay, strokes_overlay_alpha, res, 1 - strokes_overlay_alpha, 0)
        
        #length2 = max(0, length2-3) + 3
        #cv2.line(res2, (x, y), (int(x+length2*math.cos(-angle2)), int(y+length2*math.sin(-angle2))), color, 3)


cv2.imwrite(result_path, res)

cv2.imwrite(grad_dir_path, res2)
