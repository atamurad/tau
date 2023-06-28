from textpathgen import TextPathGen, DefaultParams
from renderer import TapeImageRenderer, CurveRenderer
import argparse


parser = argparse.ArgumentParser(
                    prog='tau.py',
                    description='Generate calligraphic image from input shapes and text',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input_file', help='Path to the input image file (must be white drawing on black background)')
parser.add_argument('-t', '--text', help='Text to render/fill shapes with')
parser.add_argument('-f', '--font', help='Font to use (TrueType, .ttf)')
parser.add_argument('-r', '--renderer', help='Renderer to use', choices=['raster', 'vector'])

parser.add_argument('--skel_dist', type=float, default=10.0, help='Distance threshold between contour and skeleton')

parser.add_argument('--skel_width', type=float, default=2.0, help='Skeleton width/thickness')

parser.add_argument('--dtw_spacing', type=int, default=1, help='Spacing between points when dividing contours for DTW alignment.\nWider/higher spacing runs faster but output image is less accurate.')
parser.add_argument('-o', '--output', help='Output image file')

args = parser.parse_args()

params = DefaultParams()
params.input_file = args.input_file
params.text = args.text

if args.font is not None:
    params.font = args.font

params.skel_dist_threshold = args.skel_dist
params.skel_width = args.skel_width
params.dtw_spacing = args.dtw_spacing

params.textgen = TextPathGen(params)

if args.output is not None:
    params.output = args.output

if args.renderer == "vector":
    params.renderer = CurveRenderer(params)
else:
    params.renderer = TapeImageRenderer(params)

params.textgen.process()
params.renderer.final()
