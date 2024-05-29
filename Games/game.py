import random
import time
import copy
import numpy as np

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move.

        Args:
            state (list of lists): current state of the game. This should not be modified.

        Return:
            move (list): a list of move tuples [(row, col), (source_row, source_col)] or [(row, col)]
        """
        drop_phase = True   # Initial assumption
        num_black = sum(row.count('b') for row in state)
        num_red = sum(row.count('r') for row in state)

        # Check if drop phase is over
        if num_black >= 4 and num_red >= 4:
            drop_phase = False

        if drop_phase:
            # Drop phase move selection
            move = []
            best_value, best_state = self.max_value(state, 0)  # Calculate best move using Minimax
            difference = np.array(state) != np.array(best_state)
            move_indices = np.where(difference)  # Find the index where the new piece was added

            row, col = move_indices[0][0], move_indices[1][0]
            # Ensure that we place a piece in an empty spot
            while state[row][col] != ' ':
                row, col = np.random.choice(move_indices[0]), np.random.choice(move_indices[1])

            move.append((int(row), int(col)))
            return move
        else:
            # Move phase selection
            move = []
            _, best_state_after_move = self.max_value(state, 0)
            difference = np.array(state) != np.array(best_state_after_move)
            move_indices = np.where(difference)

            # Determine source and destination of the move
            if state[move_indices[0][0]][move_indices[1][0]] == ' ':
                # The first index is the destination where the piece will move
                dest_row, dest_col = move_indices[0][0], move_indices[1][0]
                src_row, src_col = move_indices[0][1], move_indices[1][1]
            else:
                # The first index is the source where the piece is coming from
                src_row, src_col = move_indices[0][0], move_indices[1][0]
                dest_row, dest_col = move_indices[0][1], move_indices[1][1]

            # Order of the move list should be [(dest_row, dest_col), (src_row, src_col)]
            move.append((int(dest_row), int(dest_col)))
            move.append((int(src_row), int(src_col)))

            return move

    def succ(self, state, piece):
        """ Takes in a board state and returns a list of the legal successors. """
        result = []

        # Determine if it's the drop phase
        drop_phase = sum(row.count(piece) for row in state) < 4

        if drop_phase:
            # Generate successors for the drop phase
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        successor = copy.deepcopy(state)
                        successor[row][col] = piece
                        result.append(successor)
            return result
        else:
            # Generate successors for the moving phase
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            def in_bounds(x, y):
                return 0 <= x < 5 and 0 <= y < 5

            for row in range(5):
                for col in range(5):
                    if state[row][col] == piece:
                        for dx, dy in directions:
                            new_row, new_col = row + dx, col + dy
                            if in_bounds(new_row, new_col) and state[new_row][new_col] == ' ':
                                successor = copy.deepcopy(state)
                                successor[new_row][new_col] = successor[row][col]
                                successor[row][col] = ' '
                                result.append(successor)
            return result


    def heuristic_game_value(self, state, piece):
        """ Evaluates non-terminal game states and returns a floating-point value between -1 and 1. """
        # Check if the game has reached a conclusive state
        terminal_state_value = self.game_value(state)
        if terminal_state_value != 0:
            return terminal_state_value
        
        # Determine the player's piece and the opponent's piece
        player_piece, opponent_piece = ('b', 'r') if piece == 'b' else ('r', 'b')
        max_alignment_scores = {player_piece: 0, opponent_piece: 0}

        def update_max_alignment_score(alignment, player):
            """ Update the maximum score for a player based on the number of pieces aligned. """
            alignment_score = sum(alignment)
            max_alignment_scores[player] = max(max_alignment_scores[player], alignment_score)

        # Evaluate rows and columns for maximum alignments
        for index in range(5):
            update_max_alignment_score([state[index][j] == player_piece for j in range(5)], player_piece)
            update_max_alignment_score([state[index][j] == opponent_piece for j in range(5)], opponent_piece)
            update_max_alignment_score([state[j][index] == player_piece for j in range(5)], player_piece)
            update_max_alignment_score([state[j][index] == opponent_piece for j in range(5)], opponent_piece)

        # Evaluate primary and secondary diagonals
        for start in range(2):  # Only two starting points are valid for 4-length diagonals
            update_max_alignment_score([state[start + i][start + i] == player_piece for i in range(4)], player_piece)
            update_max_alignment_score([state[start + i][start + i] == opponent_piece for i in range(4)], opponent_piece)
            update_max_alignment_score([state[start + i][4 - start - i] == player_piece for i in range(4)], player_piece)
            update_max_alignment_score([state[start + i][4 - start - i] == opponent_piece for i in range(4)], opponent_piece)

        # Evaluate 2x2 box alignments
        for row in range(4):
            for col in range(4):
                update_max_alignment_score([state[row + i][col + j] == player_piece for i in range(2) for j in range(2)], player_piece)
                update_max_alignment_score([state[row + i][col + j] == opponent_piece for i in range(2) for j in range(2)], opponent_piece)

        # Normalize and return the heuristic value based on the maximum alignment scores
        if max_alignment_scores[player_piece] == max_alignment_scores[opponent_piece]:
            return 0.0
        elif max_alignment_scores[player_piece] > max_alignment_scores[opponent_piece]:
            return max_alignment_scores[player_piece] / 5.0  # Normalized score
        else:
            return -max_alignment_scores[opponent_piece] / 5.0  # Negative for opponent's advantage


    def max_value(self, state, depth):
        terminal_value = self.game_value(state)
        if terminal_value != 0:
            return terminal_value, state
        if depth >= 3:
            return self.heuristic_game_value(state, self.my_piece), state

        best_state = state
        max_value = float('-Inf')
        for successor in self.succ(state, self.my_piece):
            value, _ = self.min_value(successor, depth + 1)
            if value > max_value:
                max_value, best_state = value, successor
        return max_value, best_state

    def min_value(self, state, depth):
        terminal_value = self.game_value(state)
        if terminal_value != 0:
            return terminal_value, state
        if depth >= 3:
            return self.heuristic_game_value(state, self.my_piece), state

        best_state = state
        min_value = float('Inf')
        for successor in self.succ(state, self.my_piece):
            value, _ = self.max_value(successor, depth + 1)
            if value < min_value:
                min_value, best_state = value, successor
        return min_value, best_state


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1
        
        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        # TODO: check / diagonal wins
        for row in range(3, 5):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row-1][col+1] == state[row-2][col+2] == state[row-3][col+3]:
                    return 1 if state[i][col] == self.my_piece else-1

        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[i][col] == self.my_piece else-1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()