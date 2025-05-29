import cv2
import mediapipe as mp
import numpy as np

# === INICIALIZACIÓN DE MEDIAPIPE ===
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# === VARIABLES DEL BOTÓN DE SALIR ===
salir = False
boton_coords = (10, 400, 150, 450)  # (x1, y1, x2, y2)

def click_event(event, x, y, flags, param):
    global salir
    x1, y1, x2, y2 = boton_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if x1 <= x <= x2 and y1 <= y <= y2:
            salir = True

# === FUNCIONES AUXILIARES PARA DETECTAR GESTOS ===

def pulgar_estirado(hand_landmarks, handedness):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    return (
        abs(thumb_tip.x - thumb_mcp.x) > 0.06 and
        thumb_tip.y < thumb_ip.y
    )

def detectar_numero_cero(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distancia = np.linalg.norm([
        thumb_tip.x - index_tip.x,
        thumb_tip.y - index_tip.y
    ])
    return distancia < 0.035

def detectar_numero_uno(hand_landmarks):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )

def detectar_numero_dos(hand_landmarks):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )

def detectar_numero_tres(hand_landmarks, handedness):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
        pulgar_estirado(hand_landmarks, handedness) and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )

def detectar_numero_cuatro(hand_landmarks, handedness):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and
        not pulgar_estirado(hand_landmarks, handedness)
    )

def detectar_numero_cinco(hand_landmarks, handedness):
    return (
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and
        pulgar_estirado(hand_landmarks, handedness)
    )

def detectar_numero_seis(hand_landmarks):
    return np.linalg.norm([
        hand_landmarks.landmark[4].x - hand_landmarks.landmark[20].x,
        hand_landmarks.landmark[4].y - hand_landmarks.landmark[20].y
    ]) < 0.05

def detectar_numero_siete(hand_landmarks):
    return np.linalg.norm([
        hand_landmarks.landmark[4].x - hand_landmarks.landmark[16].x,
        hand_landmarks.landmark[4].y - hand_landmarks.landmark[16].y
    ]) < 0.05

def detectar_numero_ocho(hand_landmarks):
    return np.linalg.norm([
        hand_landmarks.landmark[4].x - hand_landmarks.landmark[12].x,
        hand_landmarks.landmark[4].y - hand_landmarks.landmark[12].y
    ]) < 0.05

def detectar_numero_nueve(hand_landmarks):
    return np.linalg.norm([
        hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x,
        hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y
    ]) < 0.05

def detectar_numero_diez(hand_landmarks, handedness):
    pulgar_arriba = pulgar_estirado(hand_landmarks, handedness)
    otros_dedos_doblados = (
        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )
    return pulgar_arriba and otros_dedos_doblados

# === CAPTURA EN TIEMPO REAL ===

cap = cv2.VideoCapture(0)
cv2.namedWindow('Numero')
cv2.setMouseCallback('Numero', click_event)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        gesture = "Detectando..."

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness_label = hand_handedness.classification[0].label

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2))

                if detectar_numero_cero(hand_landmarks):
                    gesture = "Numero 0"
                elif detectar_numero_uno(hand_landmarks):
                    gesture = "Numero 1"
                elif detectar_numero_tres(hand_landmarks, handedness_label):
                    gesture = "Numero 3"
                elif detectar_numero_dos(hand_landmarks):
                    gesture = "Numero 2"
                elif detectar_numero_cuatro(hand_landmarks, handedness_label):
                    gesture = "Numero 4"
                elif detectar_numero_cinco(hand_landmarks, handedness_label):
                    gesture = "Numero 5"
                elif detectar_numero_seis(hand_landmarks):
                    gesture = "Numero 6"
                elif detectar_numero_siete(hand_landmarks):
                    gesture = "Numero 7"
                elif detectar_numero_ocho(hand_landmarks):
                    gesture = "Numero 8"
                elif detectar_numero_nueve(hand_landmarks):
                    gesture = "Numero 9"
                elif detectar_numero_diez(hand_landmarks, handedness_label):
                    gesture = "Numero 10"
                else:
                    gesture = "No detectado"

        # Dibujar fondo negro semitransparente para el texto
        text_size, _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        text_w, text_h = text_size
        x, y = 10, 50
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # Dibujar texto en rosado
        cv2.putText(image, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 255), 2, cv2.LINE_AA)

        # Dibujar botón "Salir"
        x1, y1, x2, y2 = boton_coords
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.putText(image, "Salir", (x1 + 25, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

        # Mostrar ventana
        cv2.imshow('Numero', image)

        # Salir con ESC o con botón
        if cv2.waitKey(5) & 0xFF == 27 or salir:
            break

cap.release()
cv2.destroyAllWindows()
