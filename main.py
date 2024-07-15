import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Global variables
num_squares = 7
square_size = 80
square_positions = [(50 + i * 120, 100) for i in range(num_squares)]  # Initial positions
selected_square_idx = None
initial_square_positions = square_positions.copy()

# Function to check if the index finger is selecting a square
def check_select_square(hand_landmarks):
    global selected_square_idx

    # Extract index finger tip landmark (landmark[8])
    if hand_landmarks.landmark and len(hand_landmarks.landmark) > 8:
        index_finger_tip = hand_landmarks.landmark[8]

        finger_x = int(index_finger_tip.x * frame_width)
        finger_y = int(index_finger_tip.y * frame_height)

        for i, (sx, sy) in enumerate(square_positions):
            if sx <= finger_x <= sx + square_size and sy <= finger_y <= sy + square_size:
                selected_square_idx = i
                return True

    return False

# Function to drag the selected square
def drag_square(hand_landmarks):
    global selected_square_idx, square_positions

    # Extract index finger tip landmark (landmark[8])
    if hand_landmarks.landmark and len(hand_landmarks.landmark) > 8:
        index_finger_tip = hand_landmarks.landmark[8]

        finger_x = int(index_finger_tip.x * frame_width)
        finger_y = int(index_finger_tip.y * frame_height)
        square_positions[selected_square_idx] = (finger_x - square_size // 2, finger_y - square_size // 2)

# Function to reset square positions
def reset_square_positions():
    global square_positions
    square_positions = initial_square_positions.copy()

# Function to draw all squares on the frame
def draw_squares(frame):
    for i, (sx, sy) in enumerate(square_positions):
        cv2.rectangle(frame, (sx, sy), (sx + square_size, sy + square_size), (255, 0, 0), -1)
        cv2.putText(frame, f'Square {i+1}', (sx + 5, sy + square_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Main function to handle webcam capture and interaction
def main():
    global frame_width, frame_height

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get hand landmarks
        results = hands.process(rgb_frame)

        # Draw squares on the frame
        draw_squares(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    finger_x = int(landmark.x * frame_width)
                    finger_y = int(landmark.y * frame_height)
                    cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1)

                if check_select_square(hand_landmarks):
                    drag_square(hand_landmarks)

        # Display the resulting frame
        cv2.imshow('Drag and Drop Squares', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Reset square positions on 'r' press
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            reset_square_positions()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
