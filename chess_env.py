import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        """Инициализация шахматной среды."""
        self.board = chess.Board()

    def reset(self):
        """Сбросить шахматную доску в начальное положение."""
        self.board.reset()
        return self.get_observation()

    def get_observation(self):
        """Получить текущее состояние доски в виде 3D массива numpy."""
        obs = np.zeros((8, 8, 12), dtype=np.uint8)
        for square, piece in self.board.piece_map().items():
            row, col = divmod(square, 8)
            plane = self._piece_to_plane(piece)
            obs[row, col, plane] = 1
        return obs

    def _piece_to_plane(self, piece):
        """Определить, на какой плоскости в массиве должна быть фигура."""
        piece_type = piece.piece_type
        color = piece.color
        # Сопоставление типа и цвета фигуры с соответствующей плоскостью
        mapping = {
            (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, 
            (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11
        }
        return mapping.get((piece_type, color), -1)  # Если фигура не найдена, возвращаем -1

    def legal_actions(self):
        """Возвращает список легальных ходов в формате UCI."""
        return [move.uci() for move in self.board.legal_moves]

    def step(self, move_uci):
        """Применить ход к доске и вернуть новое состояние."""
        move = chess.Move.from_uci(move_uci)
        if move not in self.board.legal_moves:
            return self.get_observation(), -1, True, {'reason': 'illegal move'}
        
        self.board.push(move)
        done = self.board.is_game_over()
        reward = self._get_reward(done)
        return self.get_observation(), reward, done, {}

    def _get_reward(self, done):
        """Определить вознаграждение в зависимости от результата игры."""
        if not done:
            return 0
        result = self.board.result()
        if result == '1-0':
            return 1  # Победа белых
        elif result == '0-1':
            return -1  # Победа черных
        return 0  # Ничья

    def render(self):
        """Вывести текущее состояние доски."""
        print(self.board)