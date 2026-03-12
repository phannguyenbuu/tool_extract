from tool_extract.rectpack import newPacker

def pack_with_rectpack(group_patches, max_width=2954, padding=30, gap=20):
    # group_patches: [(gid, regs, orig_w, orig_h)]
    rectangles = []
    for gid, regs, orig_w, orig_h in group_patches:
        # rectpack dung (w, h, rid)
        rectangles.append((orig_w + gap, orig_h + gap, gid))

    packer = newPacker()

    # 1 bin rong max_width, cao rat lon (cho dang strip)
    BIN_W = max_width - 2 * padding
    BIN_H = 100000
    packer.add_bin(BIN_W, BIN_H)

    for w, h, rid in rectangles:
        packer.add_rect(w, h, rid=rid)

    packer.pack()

    # map gid -> (x_off, y_off, w, h)
    placements = {}
    for b, x, y, w, h, rid in packer.rect_list():
        x_off = x + padding
        y_off = y + padding
        placements[rid] = (x_off, y_off, w, h)

    return placements, BIN_W
