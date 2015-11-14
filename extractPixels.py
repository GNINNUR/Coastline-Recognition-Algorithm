from PIL import Image
im = Image.open('cannyEdgeImageDenoisingBilateralFilterColorFilterImg.jpg')

rgb_im = im.convert('RGB')

coastlinePoints = []

for x in range(0, im.size[0]):
    for y in range(0, im.size[1]):
        r, g, b = rgb_im.getpixel((x, y))
        
        if((r ==255) and (b ==255) and (g ==255)):
            pixelPosition = []
            pixelPosition.append(x)
            pixelPosition.append(y)
            coastlinePoints.append(pixelPosition)
            
            print coastlinePoints



print im.size

