import cv2
import numpy as np

cap = cv2.VideoCapture("q1/q1B.mp4")

lower_orange = np.array([3, 50, 180])
upper_orange = np.array([23, 255, 255])

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

last_blue_rect = None

## AS LINHAS COMENTADAS SÃO FILTROS QUE DEIXAM A DETECÇÃO 100% PRECISA, MAS DEIXAM A REDERIZAÇÃO SUPER LENTA

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv = cv2.medianBlur(hsv, 7)

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    #kernel = np.ones((5, 5), np.uint8)
    #mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)

    #mask_orange = cv2.dilate(mask_orange, kernel, iterations=2)

    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []

    current_orange_rect = None
    for contour in contours_orange:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            shapes.append((area, x, y, w, h, 'orange'))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            current_orange_rect = (x, y, w, h)

    current_blue_rect = None
    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            shapes.append((area, x, y, w, h, 'blue'))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            current_blue_rect = (x, y, w, h)

    if current_blue_rect:
        last_blue_rect = current_blue_rect

    if last_blue_rect and not current_blue_rect:
        x, y, w, h = last_blue_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if shapes:
        max_shape = max(shapes, key=lambda s: s[0])
        _, x, y, w, h, color = max_shape
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    status_text = "STANDBY"

    orange_shape = next((s for s in shapes if s[5] == 'orange'), None)
    blue_shape = next((s for s in shapes if s[5] == 'blue'), None)

    if orange_shape and blue_shape:
        ox, oy, ow, oh = orange_shape[1:5]
        bx, by, bw, bh = blue_shape[1:5]

        if ox < bx + bw and ox + ow > bx and oy < by + bh and oy + oh > by:
            status_text = "COLISÃO DETECTADA"
        elif ox + ow > bx:
            status_text = "BARREIRA PASSADA"

    cv2.putText(frame, status_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
