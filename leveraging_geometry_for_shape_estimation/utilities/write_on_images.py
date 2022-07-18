import cv2

def draw_text_block(img,text,top_left_corner=(20,20),font_scale=3,font_thickness=2):

    line_height = 20 * font_scale

    for i,line in enumerate(text):
        pos = (top_left_corner[0],top_left_corner[1] + (i + 1) *line_height)
        draw_text(img,line,pos=pos,font_scale=font_scale,font_thickness=font_thickness)


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def draw_lines(input_img,lines_2D,draw_indices=False):


    for i,line in enumerate(lines_2D):
        y_start, x_start, y_end, x_end = [int(val) for val in line]
        cv2.line(input_img, (x_start, y_start), (x_end, y_end), [255,0,0], 2)
        if draw_indices:
            cv2.putText(input_img, str(i), tuple([int((x_start + x_end)/2),int((y_start + y_end)/2)]), cv2.FONT_HERSHEY_SIMPLEX, 3,(255, 0, 0), 2, cv2.LINE_AA)

    return input_img