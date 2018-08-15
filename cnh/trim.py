from PIL import Image, ImageChops

im = Image.open('d.jpg')

def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -70)
        bbox = diff.getbbox()
        if bbox:
                return im.crop(bbox)


trim(im).show()