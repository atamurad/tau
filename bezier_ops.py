import bezier
from freetype import Face
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform


#
# Import glyph as curved polygon
#
def polygon_load_glyph(font, size, char):
    face = Face(font)
    face.set_char_size(size*64)
    face.load_char(char)
    slot = face.glyph

    outline = slot.outline
    points = np.array(outline.points, dtype=[('x', float), ('y', float)])
    start, end = 0, 0

    res = []

    # Iterate over each contour
    for i in range(len(outline.contours)):
        end = outline.contours[i]
        points = outline.points[start:end+1]
        points.append(points[0])
        tags = outline.tags[start:end+1]
        tags.append(tags[0])

        segments = [[points[0], ], ]
        for j in range(1, len(points)):
            segments[-1].append(points[j])
            if tags[j] & (1 << 0) and j < (len(points)-1):
                segments.append([points[j], ])

        edges = []
        verts = [points[0], ]
        for segment in segments:
            if len(segment) == 2:
                verts.extend(segment)
                edges.append(bezier.Curve(np.asfortranarray(verts[-2:]).T, degree=1))
            elif len(segment) == 3:
                verts.extend(segment)
                edges.append(bezier.Curve(np.asfortranarray(verts[-3:]).T, degree=2))
            else:
                verts.append(segment[0])
                verts.append(segment[1])
                for i in range(1, len(segment)-2):
                    A, B = segment[i], segment[i+1]
                    C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                    verts.extend([C, B])
                    edges.append(bezier.Curve(np.asfortranarray(verts[-4:-1]).T, degree=2))
                verts.append(segment[-1])
                edges.append(bezier.Curve(np.asfortranarray(verts[-3:]).T, degree=2))

        start = end+1

        pol = bezier.CurvedPolygon(*edges)
        res.append(pol)
    return res, slot.advance.x

#
# Slice CurvedPolygon at x1 vertical line
#
def polygon_slice(polygon, x1, ax=None):

    # construct vertical lines at x1 and x2
    min_y = 0
    max_y = 0
    for e in polygon._edges:
        min_y = min(min_y, e.nodes[1, :].min())
        max_y = max(max_y, e.nodes[1, :].max())
    min_y -= 100.0
    max_y += 100.0

    x1line = bezier.Curve(np.asfortranarray([
                [x1, x1],
                [min_y, max_y]
            ]), degree=1)

    if ax:
        polygon.plot(25, ax=ax)
        x1line.plot(25, ax=ax)

    # find and plot intersections
    # cuts intersecting edges to 2 sides
    edges = []
    iscs = []

    # these are field index numbers in a tuple - refactor later to struct/dict
    CUT1 = 2
    CUT2 = 3

    for e in polygon._edges:

        intersections = e.intersect(x1line)
        s_vals = np.asfortranarray(intersections[0, :])

        # for now we assume curve intersects the vertical line at most once
        # but cubic bezier curves can intersect at 3 points.
        # if len(s_vals) > 1:
            # print(s_vals)
            # print(e.nodes)

        if len(s_vals) == 1:

            e_left = e.specialize(0.0, s_vals[0])
            e_right = e.specialize(s_vals[0], 1.0)

            points = np.array(e.evaluate_multi(s_vals))

            if ax:
                ax.scatter(points[0, :], points[1, :])

            cut1 = dict()
            cut1["p1"] = (points[0, 0], points[1, 0])
            cut1["p2"] = None

            cut2 = dict()
            cut2["p1"] = None
            cut2["p2"] = (points[0, 0], points[1, 0])
            cut2["next"] = e_right

            iscs.append((points[0, 0], points[1, 0], cut1, cut2))

            edges.append(e_left)
            edges.append(cut1)
            edges.append(cut2)
            edges.append(e_right)
        else:
            edges.append(e)

    # sort intersections by y coordinates
    if len(iscs) % 2 != 0:
        for i in iscs:
            print(i)
        return [], []
    assert(len(iscs) % 2 == 0)
    iscs = sorted(iscs, key=lambda x: x[1])

    # join pairs on both sides
    for i in range(0, len(iscs), 2):
        if ax:
            ax.vlines(x1, iscs[i][1], iscs[i+1][1], colors="black")

        # link i with i+1
        iscs[i][CUT1]["p2"] = (iscs[i+1][0], iscs[i+1][1])
        iscs[i][CUT1]["next"] = iscs[i+1][CUT2]["next"]
        iscs[i][CUT2]["p1"] = (iscs[i+1][0], iscs[i+1][1])

        iscs[i+1][CUT1]["p2"] = (iscs[i][0], iscs[i][1])
        iscs[i+1][CUT1]["next"] = iscs[i][CUT2]["next"]
        iscs[i+1][CUT2]["p1"] = (iscs[i][0], iscs[i][1])

    # convert cycles to polygons
    used = []
    pols_left = []
    pols_right = []

    while True:
        # find first unused edge
        start = None
        for i in edges:
            if i not in used and type(i) != dict:
                start = i
                break
        if start is None:
            break

        side = start.nodes.T[1][0] > x1

        # follow cycle
        cur = start
        pol = []
        while True:
            used.append(cur)
            if type(cur) == dict:
                pol.append(bezier.Curve(np.asfortranarray([cur["p1"], cur["p2"]]).T, degree=1))
                cur = cur["next"]
            else:
                pol.append(cur)
                cur = edges[(edges.index(cur)+1) % len(edges)]
            if cur == start:
                break

        pol = bezier.CurvedPolygon(*pol)

        if side:
            pols_right.append(pol)
            if ax:
                pol.plot(100, ax=ax, color="red")
        else:
            pols_left.append(pol)
            if ax:
                pol.plot(100, ax=ax, color="green")

    return pols_left, pols_right


#
# Crop CurvedPolygon between x1 and x2 vertial lines
#
def polygon_crop(polygons, x1, x2):
    tmp = []
    # take right side of x1
    for cpol in polygons:
        left, right = polygon_slice(cpol, x1)
        tmp.extend(right)

    # take left side of x2
    res = []
    for cpol in tmp:
        left, right = polygon_slice(cpol, x2)
        res.extend(left)
    # shift output to left
    m = transform.AffineTransform(translation=(-x1, 0.)).params
    res = polygon_transform(res, m)
    return res


#
# Transform CurvedPolygon
#
def polygon_transform(polygons, M):
    res = []
    for cpol in polygons:
        pol = []
        for e in cpol._edges:
            nodes = []
            for n in e.nodes.T:
                p = (n[0], n[1], 1.0)
                p = M @ p
                nodes.append([p[0]/p[2], p[1]/p[2]])
            pol.append(bezier.Curve(np.asfortranarray(nodes).T, e.degree))
        res.append(bezier.CurvedPolygon(*pol, _verify=False))
    return res


#
# Export CurvedPolygons as SVG file
#
def polygon_export(polygons, filename):

    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    for polygon in polygons:
        for e in polygon._edges:
            min_y = min(min_y, e.nodes[1, :].min())
            max_y = max(max_y, e.nodes[1, :].max())
            min_x = min(min_x, e.nodes[0, :].min())
            max_x = max(max_x, e.nodes[0, :].max())

    svg = f'<svg width="{max_x-min_x}" height="{max_y-min_y}" viewBox="{min_x} {min_y} {max_x-min_x} {max_y-min_y}" xmlns="http://www.w3.org/2000/svg">'
    svg += '<g>'
    svg += "<path fill-rule=\"nonzero\" d=\""
    for pol in polygons:
        start = pol._edges[0].nodes.T[0]
        svg += f"\nM {start[0]} {start[1]} "
        for edge in pol._edges:
            nodes = edge.nodes.T[1:]
            if len(nodes) == 1:
                svg += f"L {nodes[0][0]} {nodes[0][1]} "
            elif len(nodes) == 2:
                svg += f"Q {nodes[0][0]} {nodes[0][1]}, {nodes[1][0]} {nodes[1][1]} "
            elif len(nodes) == 3:
                svg += f"C {nodes[0][0]} {nodes[0][1]}, {nodes[1][0]} {nodes[1][1]}, {nodes[2][0]} {nodes[2][1]} "
            else:
                print("ERROR: Higher order curve not supported by SVG export!")
    # close the loop
    svg += "\" fill='black'/>\n"
    svg += '</g></svg>'

    with open(filename, "w") as fout:
        fout.write(svg)


def maketext(font, text):
    pols = []
    pos = 0.0
    for c in text:
        g, advance_x = polygon_load_glyph(font, 16, c)
        m = transform.AffineTransform(translation=(pos, 16.), scale=(1.0/64, -1.0/64)).params
        pols.extend(polygon_transform(g, m))
        pos += advance_x/64.0
    return pols


tau = (
    "6.28318530717958647692528676655900576839433879875021164194988918461563281257241799725606965068423413596429617302656461329418768921910116446345"
    # + "07188162569622349005682054038770422111192892458979098607639288576219513318668922569512964675735663305424038182912971338469206972209086532964267872145204982825474491740132"
)


if __name__ == "__main__":
    # glyph, _ = polygon_load_glyph('./Ubuntu-B.ttf', 48, 'G')
    glyph = maketext("QUICKBROWNFOXJUMPEDOVERTHELAZYDOG")
    # glyph = polygon_crop(glyph, 0, 10.0)

    """
    m = transform.AffineTransform(rotation=-5,
                                  translation=(250., 250.),
                                  scale=(0.1, -0.1)).params
    glyph = polygon_transform(glyph, m)
    """

    polygon_export(glyph, "bezier.svg")
