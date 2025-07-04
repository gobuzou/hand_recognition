import cv2
import mediapipe as mp
import PySimpleGUI as sg
import numpy as np
import copy
import random

# --- オセロの基本設定 ---
EMPTY = 0
BLACK = -1
WHITE = 1
BOARD_SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1), (1, 0), (1, 1)]

# --- ミニマックス用評価関数 ---
def evaluate(board, color):
    return np.sum(board) * color

# --- 合法手チェック ---
def get_valid_moves(board, color):
    valid_moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board[y][x] == EMPTY and can_place(board, x, y, color):
                valid_moves.append((x, y))
    return valid_moves

def can_place(board, x, y, color):
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        flipped = False
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[ny][nx] == -color:
                flipped = True
            elif board[ny][nx] == color:
                if flipped:
                    return True
                break
            else:
                break
            nx += dx
            ny += dy
    return False

# --- 石を置く ---
def apply_move(board, x, y, color):
    board[y][x] = color
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        stones = []
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[ny][nx] == -color:
                stones.append((nx, ny))
            elif board[ny][nx] == color:
                for sx, sy in stones:
                    board[sy][sx] = color
                break  # ✅ この break はこの方向だけに適用
            else:
                break
            nx += dx
            ny += dy


# --- ミニマックス ---
def minimax(board, depth, color, maximizing):
    moves = get_valid_moves(board, color)
    if depth == 0 or not moves:
        return evaluate(board, color), None

    best_move = None
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            b_copy = copy.deepcopy(board)
            apply_move(b_copy, move[0], move[1], color)
            eval, _ = minimax(b_copy, depth-1, -color, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            b_copy = copy.deepcopy(board)
            apply_move(b_copy, move[0], move[1], color)
            eval, _ = minimax(b_copy, depth-1, -color, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

# --- 盤面初期化 ---
def init_board():
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board

# --- 描画処理 ---
def draw_board(image, board):
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            px = x * CELL_W + CELL_W // 2
            py = y * CELL_H + CELL_H // 2
            if board[y][x] == WHITE:
                cv2.circle(image, (px, py), RADIUS, (255,255,255), -1)
            elif board[y][x] == BLACK:
                cv2.circle(image, (px, py), RADIUS, (0,0,0), -1)
    for i in range(1, BOARD_SIZE):
        cv2.line(image, (0, i*CELL_H), (IMG_W, i*CELL_H), (0,255,0), 1)
        cv2.line(image, (i*CELL_W, 0), (i*CELL_W, IMG_H), (0,255,0), 1)

# --- メイン処理 ---
IMG_W, IMG_H = 1200, 600
CELL_W = IMG_W // BOARD_SIZE
CELL_H = IMG_H // BOARD_SIZE
RADIUS = min(CELL_W, CELL_H) // 2 - 5

sg.theme('DarkBlue')
layout = [
    [sg.Text('あなたが白（先手）です。'), sg.Button('決定', key='choice')],
    [sg.Image(filename='', key='board')],
    [sg.Button('終了', key='end')]
]
window = sg.Window('オセロ対戦', layout, finalize=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

board = init_board()
turn = WHITE
index_pos = None

while True:
    event, _ = window.read(timeout=50)
    if event == sg.WIN_CLOSED or event == 'end':
        break

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (IMG_W, IMG_H))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    image = frame.copy()
    
    index_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[8]  # 人差し指の先端
            cx, cy = int(tip.x * IMG_W), int(tip.y * IMG_H)
            index_pos = (cx, cy)
            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)

    # プレイヤーのターン
    if event == 'choice' and turn == WHITE and index_pos:
        col = index_pos[0] // CELL_W
        row = index_pos[1] // CELL_H
        if can_place(board, col, row, WHITE):
            apply_move(board, col, row, WHITE)
            turn = BLACK

    # AIのターン
    if turn == BLACK:
        valid = get_valid_moves(board, BLACK)
        if valid:
            _, move = minimax(board, 3, BLACK, True)
            if move:
                apply_move(board, move[0], move[1], BLACK)
        turn = WHITE

    # パス処理
    if not get_valid_moves(board, turn):
        turn *= -1
        if not get_valid_moves(board, turn):
            white_count = sum(row.count(WHITE) for row in board)
            black_count = sum(row.count(BLACK) for row in board)
            winner = '引き分け'
            if white_count > black_count:
                winner = 'あなたの勝ち！'
            elif black_count > white_count:
                winner = 'AIの勝ち！'
            sg.popup(f'終了！\n白: {white_count} 黒: {black_count}\n{winner}')
            break

    draw_board(image, board)
    imgbytes = cv2.imencode('.png', image)[1].tobytes()
    window['board'].update(data=imgbytes)

cap.release()
hands.close()
window.close()