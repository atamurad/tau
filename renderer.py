from PIL import Image, ImageDraw, ImageFont
from skimage import transform
from shapely import Polygon, MultiPolygon, union_all
import numpy as np
from bezier_ops import maketext, polygon_export, polygon_slice, polygon_transform


class TapeImageRenderer:

    def __init__(self, params):

        # create output image
        self.out = Image.new("RGBA", params.input.shape, (0, 0, 0, 255))
        self.out_fname = params.output
        self.shape = params.input.shape
        self.txt = params.text

        # render text image
        self.height = params.font_size
        self.txtout = Image.new("RGBA", (2048, self.height), (255, 255, 255, 0))
        self.fnt = ImageFont.truetype(params.font, self.height)
        self.d = ImageDraw.Draw(self.txtout)
        self.d.text(
            (0, 0),
            self.txt,
            font=self.fnt,
            fill=(0, 255, 0, 255),
        )

    def rewind(self):
        self.pos = 0

    def advance(self, step_w, transform):
        crop = self.txtout.crop((self.pos, 0, self.pos + step_w, self.height))

        if np.linalg.matrix_rank(transform) != 3:
            print("not invertible....")
            return

        transform = np.linalg.inv(transform)

        patch = crop.transform(
            self.shape,
            method=Image.PERSPECTIVE,
            data=transform.reshape(9),
            resample=Image.Resampling.BICUBIC,
        )

        self.out = Image.alpha_composite(self.out, patch)
        self.pos += step_w

    def final(self):
        self.out.save(self.out_fname)
        print(f"Output image saved {self.out_fname}")


class SVGRenderer:
    def __init__(self, params):
        self.shape = params.input.shape
        self.height = 16
        self.txtout = SVGRenderer.__text2polygon(params.text, params.font, self.height)
        self.out = []
        self.out_fname = params.output

    def __text2polygon(text, font, h):

        def translate(pol, x, y):
            shell = []
            holes = []
            for p in pol.exterior.coords:
                shell.append((p[0] / 64.0 + x, -p[1] / 64.0 + y))
            for hole in pol.interiors:
                h = []
                for p in hole.coords:
                    h.append((p[0] / 64.0 + x, -p[1] / 64.0 + y))
                holes.append(h)
            return Polygon(shell, holes)

        pos = 0.0
        letters = []

        # load font
        from freetype import FT_LOAD_DEFAULT, FT_LOAD_NO_BITMAP, Face
        face = Face(font)
        flags = FT_LOAD_DEFAULT | FT_LOAD_NO_BITMAP
        face.set_char_size(int(h) * 64)

        miny = 1e10
        maxy = 0.0

        for c in text:
            print(f"char: {c} pos: {pos}")

            face.load_char(c, flags)
            slot = face.glyph
            outline = slot.outline

            holes = []
            for i in range(1, len(outline.contours)):
                holes.append(
                    outline.points[outline.contours[i - 1] + 1 : outline.contours[i] + 1]
                )

            p = Polygon(outline.points[: outline.contours[0] + 1], holes)

            for i in outline.points:
                miny = min(miny, i[1])
                maxy = max(maxy, i[1])

            p = translate(p, pos, h)

            letters.append(p)
            pos += slot.advance.x / 64.0

        print(f"miny: {miny}")
        print(f"maxy: {maxy}")
        print(f"h: {maxy-miny}")

        txt = union_all(letters)
        return txt

    def rewind(self):
        self.pos = 0.0

    def advance(self, step_w, transform):
        def apply_transform(crop, transform):
            proj = []
            for p in crop.exterior.coords:
                p_proj = transform @ np.array([p[0] - self.pos, p[1], 1.0])
                proj.append((p_proj[0] / p_proj[2], p_proj[1] / p_proj[2]))
            holes = []
            for hole in crop.interiors:
                h = []
                for p in hole.coords:
                    p_proj = transform @ np.array([p[0] - self.pos, p[1], 1.0])
                    h.append((p_proj[0] / p_proj[2], p_proj[1] / p_proj[2]))
                holes.append(h)
            return Polygon(proj, holes)

        # if np.linalg.matrix_rank(transform) != 3:
        #    print("not invertible....")
        #    return

        # transform = np.linalg.inv(transform)

        rect = Polygon(
            [
                [self.pos, 0],
                [self.pos + step_w, 0],
                [self.pos + step_w, self.height],
                [self.pos, self.height],
            ]
        )
        crop = self.txtout.intersection(rect)
        # crop = rect

        # print("intersection: ", len(crop.exterior.coords), crop.area)
        if type(crop) == Polygon:
            crop = apply_transform(crop, transform)
        elif type(crop) == MultiPolygon:
            mp = []
            for p in crop.geoms:
                mp.append(apply_transform(p, transform))
            crop = MultiPolygon(mp)
        else:
            return

        # print(crop, crop.is_valid, crop.area)
        # if crop.is_valid:
        self.out.append(crop)
        self.pos += step_w

    def final(self):
        all = union_all(self.out, grid_size=0)
        with open(self.out_fname, "w") as f:
            f.write(
                f'<svg version="1.1" width="{self.shape[0]}" height="{self.shape[1]}" xmlns="http://www.w3.org/2000/svg">\n'
            )
            f.write(all.svg(scale_factor=0.0, opacity=1.0))
            f.write("</svg>")
        print(f"Output image saved {self.out_fname}")


class CurveRenderer:
    def __init__(self, params):
        self.height = params.font_size
        self.font = params.font
        self.txt = params.text
        self.out = []
        self.out_fname = params.output

    def rewind(self):
        self.pos = 0.0
        self.count = 0
        self.txtout = maketext(self.font, self.txt)

    def advance(self, step_w, T):
        # print(self.pos, step_w)

        left = []
        right = []
        for cpol in self.txtout:
            l, r = polygon_slice(cpol, step_w)
            left.extend(l)
            right.extend(r)

        # shift right side to left
        M = transform.AffineTransform(translation=(-step_w, 0.)).params
        self.txtout = polygon_transform(right, M)

        crop = polygon_transform(left, T)
        self.out.extend(crop)

        self.pos += step_w
        # save progress
        self.count += 1
        if self.count % 10 == 0:
            polygon_export(self.out, self.out_fname)
            print(f"Saving progress image {self.out_fname}")

    def final(self):
        polygon_export(self.out, self.out_fname)
        print(f"Final output image saved {self.out_fname}")
