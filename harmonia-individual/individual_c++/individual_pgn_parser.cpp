#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <regex>
#include <sstream>
#include <cstdlib>
#include <map>
#include <random>
#include <zstd.h>    
#include <filesystem>
#include <cstring>

#include "chess.hpp"

using namespace chess;

class MyVisitor : public pgn::Visitor {
   private:
    // Define a packed structure matching your dictionary layout.
    // (Note: For simplicity, we use float for the np.float16 fields.)
    #pragma pack(push, 1)
    struct Dictionary {
        int8_t  turn;
        int8_t  white_kingside_castling_rights;
        int8_t  white_queenside_castling_rights;
        int8_t  black_kingside_castling_rights;
        int8_t  black_queenside_castling_rights;
        int8_t  board_position[64];  // 64-element int8 array.
        int8_t  from_square;
        int8_t  to_square;
        int8_t  length;
        int8_t  phase;
        int8_t  result;
        int8_t  categorical_result;
        int16_t base_time;
        int16_t increment_time;
        float   white_remaining_time;  // using float instead of np.float16
        float   black_remaining_time;
        int16_t white_rating;
        int16_t black_rating;
        float   time_spent_on_move;
        int16_t move_number;
        int16_t num_legal_moves;
        int16_t white_material_value;
        int16_t black_material_value;
        int16_t material_difference;
        float   moves_until_end;
        char fen[200];
        char white_username[100];
        char black_username[100];
    };
    #pragma pack(pop)
    // The expected size of Dictionary should be 109 bytes.

    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> moves_with_turns;
    std::vector<std::string> white_moves;
    std::vector<std::string> black_moves;
    std::vector<std::string> moves_with_time;
    std::vector<std::string> comments;
    std::vector<std::string> turns;
    std::vector<std::string> fens;
    std::vector<double> white_clock_times;
    std::vector<double> black_clock_times;
    std::vector<double> white_elapsed_times;
    std::vector<double> black_elapsed_times;
    std::vector<int> move_numbers;
    Board clock_board;
    int& games_processed;
    int total_games;
    int games_removed;
    int white_move_count = 0;
    int black_move_count = 0;
    //int num_chunks = 0;
    int file_count = 0;
    int dir_count = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::vector<Dictionary> chunk;
    std::string output_dir;
    int chunks_per_file;
    int max_files_per_dir;

   public:
    MyVisitor(int& processed, int total, const std::string& outputdir, int chunksperfile, int maxfilesperdir)
        : games_processed(processed), total_games(total), output_dir(outputdir), chunks_per_file(chunksperfile), max_files_per_dir(maxfilesperdir) {
        start_time = std::chrono::high_resolution_clock::now();
        fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        clock_board = Board();
        games_removed = 0;

        std::filesystem::create_directory(output_dir);
        std::filesystem::create_directory(output_dir+"/"+std::to_string(dir_count));
    }

    std::vector<std::string> generateLegalMovesFromFEN(const Board& board) {

        // Create a move list to store legal moves
        Movelist moveList;

        // Generate all legal moves
        movegen::legalmoves<chess::movegen::MoveGenType::ALL>(moveList, board);

        // Convert all moves in moveList to UCI notation and store them
        std::vector<std::string> uciMoves;
        for (const Move& move : moveList) {
            uciMoves.push_back(uci::moveToUci(move));
        }

        return uciMoves;
        /*
        // Print the UCI moves
        std::cout << "Legal moves for FEN: " << fen << std::endl;
        for (const std::string& uciMove : uciMoves) {
            std::cout << uciMove << " ";
        }
        std::cout << std::endl;
        */
    }

    std::string getCastlingRights(const Board& board) {
        std::string castling = board.getCastleString();

        return castling;
    }

    std::string getColor(const Board& board) {
        Color side = board.sideToMove();

        return std::to_string(side);
    }


    std::unordered_map<std::string, int> getPieceStats(const Board& board, std::string& turn){
        int total_pieceCount = 0;
        int whitePieceCount = 0;
        int blackPieceCount = 0;
        int whiteValue = 0;
        int blackValue = 0;

        std::unordered_map<PieceType::underlying, int> pieceValues = {
            {PieceType::underlying::PAWN, 1},
            {PieceType::underlying::KNIGHT, 3},
            {PieceType::underlying::BISHOP, 3},
            {PieceType::underlying::ROOK, 5},
            {PieceType::underlying::QUEEN, 9},
            {PieceType::underlying::KING, 0} // King's value is 0
        };

        for (int sqIndex = 0; sqIndex < 64; ++sqIndex) {
            Square sq = static_cast<Square>(sqIndex);  // Convert index to Square enum
            
            // Get the piece at the current square
            Piece piece = board.at(sq);
            
            // Check if the square contains a piece (i.e., it's not an empty square)
            if (piece != Piece::NONE) {
                int pieceMaterialValue = pieceValues[piece.type().internal()];

                total_pieceCount++;
                if (piece.color() == Color::WHITE) {
                    whitePieceCount++;
                    whiteValue += pieceMaterialValue;
                } else if (piece.color() == Color::BLACK) {
                    blackPieceCount++;
                    blackValue += pieceMaterialValue;
                }
            }
        }

        int material_difference = 0;
        if(turn=="0"){
            material_difference = whiteValue - blackValue;
        }
        if(turn=="1"){
            material_difference = blackValue - whiteValue;
        }

        std::unordered_map<std::string, int> piece_stats;
        piece_stats["total_pieces"] = total_pieceCount;
        piece_stats["white_piece_count"] = whitePieceCount;
        piece_stats["black_piece_count"] = blackPieceCount;
        piece_stats["white_material_value"] = whiteValue;
        piece_stats["black_material_value"] = blackValue;
        piece_stats["material_difference"] = material_difference;

        return piece_stats;
    }

    std::string replace_number(const std::string& input) {
        std::string result = input;
        std::regex re("\\d");  // Regex to match any digit
        std::smatch match;
        
        // Loop through all the matches in the input string
        while (std::regex_search(result, match, re)) {
            int num = std::stoi(match.str());  // Get the number as an integer
            std::string replacement(num, '.');  // Create a string with 'num' periods
            result.replace(match.position(), match.length(), replacement);  // Replace the match
        }
        return result;
    }

    // Function to parse the board from the FEN string
    std::vector<int> parse_board(const std::string& fen) {
        // Split FEN by spaces and extract the board layout (before the first space)
        
        std::stringstream ss(fen);
        std::string board, turn, castling_rights, ep_square, underscore1, underscore2;
        ss >> board >> turn >> castling_rights >> ep_square >> underscore1 >> underscore2;

        
        //std::cout << ep_square << std::endl;
        board.erase(std::remove(board.begin(), board.end(), '/'), board.end());
        

        // Define the regex pattern to match digits
        std::regex digit_pattern("\\d");

        // Create a regex iterator to find all matches
        auto begin = std::sregex_iterator(board.begin(), board.end(), digit_pattern);
        auto end = std::sregex_iterator();

        // Initialize a new string to build the result
        std::string result;
        size_t last_pos = 0;

        // Iterate over all matches
        for (auto it = begin; it != end; ++it) {
            // Append the substring before the match
            result.append(board.substr(last_pos, it->position() - last_pos));
            // Append the replacement for the match
            result.append(replace_number(it->str()));
            // Update the last position
            last_pos = it->position() + it->length();
        }
        

        // Append the remaining part of the string after the last match
        result.append(board.substr(last_pos));
        board = result;
        if (ep_square != "-"){
            board = assign_ep_square(board, ep_square);
        }
        
        std::vector<int> parsed_encoded_board;
        for (char it : board) {
            parsed_encoded_board.push_back(PIECES[it]);
        }

        // Replace numbers with periods
        return parsed_encoded_board;
    }

    const std::vector<std::string> FILES = {"a", "b", "c", "d", "e", "f", "g", "h"};
    const std::vector<std::string> RANKS = {"1", "2", "3", "4", "5", "6", "7", "8"};
    std::unordered_map<char, int> PIECES = {
        {'.', 0},
        {',', 1},
        {'P', 2},
        {'p', 3},
        {'R', 4},
        {'r', 5},
        {'N', 6},
        {'n', 7},
        {'B', 8},
        {'b', 9},
        {'Q', 10},
        {'q', 11},
        {'K', 12},
        {'k', 13}
    };
    std::unordered_map<std::string, int> SQUARES = {
        {"a8", 0}, {"b8", 1}, {"c8", 2}, {"d8", 3}, {"e8", 4}, {"f8", 5}, {"g8", 6}, {"h8", 7},
        {"a7", 8}, {"b7", 9}, {"c7", 10}, {"d7", 11}, {"e7", 12}, {"f7", 13}, {"g7", 14}, {"h7", 15},
        {"a6", 16}, {"b6", 17}, {"c6", 18}, {"d6", 19}, {"e6", 20}, {"f6", 21}, {"g6", 22}, {"h6", 23},
        {"a5", 24}, {"b5", 25}, {"c5", 26}, {"d5", 27}, {"e5", 28}, {"f5", 29}, {"g5", 30}, {"h5", 31},
        {"a4", 32}, {"b4", 33}, {"c4", 34}, {"d4", 35}, {"e4", 36}, {"f4", 37}, {"g4", 38}, {"h4", 39},
        {"a3", 40}, {"b3", 41}, {"c3", 42}, {"d3", 43}, {"e3", 44}, {"f3", 45}, {"g3", 46}, {"h3", 47},
        {"a2", 48}, {"b2", 49}, {"c2", 50}, {"d2", 51}, {"e2", 52}, {"f2", 53}, {"g2", 54}, {"h2", 55},
        {"a1", 56}, {"b1", 57}, {"c1", 58}, {"d1", 59}, {"e1", 60}, {"f1", 61}, {"g1", 62}, {"h1", 63}
    };
    std::unordered_map<std::string, int> phase_encoder = {
        {"opening", 0},
        {"middlegame", 1},
        {"endgame", 2}
    };


    // Function to get the index of a square on the chessboard
    int square_index(const std::string& square) {
        // Extract the file and rank from the square notation (e.g., "e4")
        std::string file = square.substr(0, 1);  // First character (file)
        std::string rank = square.substr(1, 1);  // Second character (rank)

        // Find the index of the file and rank in their respective arrays
        int file_index = std::distance(FILES.begin(), std::find(FILES.begin(), FILES.end(), file));
        int rank_index = std::distance(RANKS.begin(), std::find(RANKS.begin(), RANKS.end(), rank));

        // Calculate the square index based on the rank (from the top of the board)
        return (7 - rank_index) * 8 + file_index;
    }

    // Function to assign the En Passant square to the board
    std::string assign_ep_square(const std::string& board, const std::string& ep_square) {
        // Get the index of the En Passant square
        int i = square_index(ep_square);

        // Create a new string to represent the board with the En Passant square
        return board.substr(0, i) + "," + board.substr(i + 1);
    }

    std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\n\r\f\v");
        size_t end = str.find_last_not_of(" \t\n\r\f\v");

        return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
    }

    double timeToSeconds(const std::string& timeStr) {
        std::stringstream ss(timeStr);
        std::vector<std::string> timeParts;
        std::string part;
    
        // Split the string by ':'
        while (std::getline(ss, part, ':')) {
            timeParts.push_back(part);
        }
    
        // Ensure the format is at least HH:MM:SS
        if (timeParts.size() != 3) {
            throw std::invalid_argument("Invalid time format! Expected HH:MM:SS or HH:MM:SS.sss");
        }
    
        // Convert HH and MM to integers
        int hours = std::stoi(timeParts[0]);
        int minutes = std::stoi(timeParts[1]);
    
        // Convert SS.sss to a floating-point number
        double seconds = std::stod(timeParts[2]);
    
        // Convert to total seconds
        return hours * 3600 + minutes * 60 + seconds;
    }

    template <typename T>
    void printArray(const std::vector<T>& vec) {
            std::cout << "[ ";
            for (size_t i = 0; i < vec.size(); i++) {
                std::cout << vec[i];
                if (i < vec.size() - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }

    std::vector<std::string> splitString(const std::string& str, char delimiter) {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            result.push_back(token);
        }

        return result;
    }

    void write_to_file(const std::vector<Dictionary>& dicts, const std::string&output_file){
        size_t numDicts = dicts.size();
        size_t totalSize = numDicts * sizeof(Dictionary);
        std::vector<char> buffer(totalSize);
        // Copy each dictionary into the buffer sequentially.
        for (size_t i = 0; i < numDicts; i++) {
            std::memcpy(buffer.data() + i * sizeof(Dictionary),
                        &dicts[i], sizeof(Dictionary));
        }

        size_t const maxCompressedSize = ZSTD_compressBound(buffer.size());
        std::vector<char> compressedBuffer(maxCompressedSize);
        size_t compressedSize = ZSTD_compress(compressedBuffer.data(), maxCompressedSize,
                                            buffer.data(), buffer.size(), 1);
        if (ZSTD_isError(compressedSize))
            throw std::runtime_error("Compression error: " + std::string(ZSTD_getErrorName(compressedSize)));
        std::ofstream outFile(output_file, std::ios::binary);
        if (!outFile)
            throw std::runtime_error("Cannot open output file: " + output_file);
        outFile.write(compressedBuffer.data(), compressedSize);
        outFile.close();
    }


    void header(std::string_view key, std::string_view value) override {
        //std::cout << "headers started" << std::endl;
        //std::cout << key << std::endl;
        //std::cout << value << std::endl;
        headers.emplace_back(key, value);
        //std::cout << "headers array created" << std::endl;
        //printArray(headers);
    }

    void move(std::string_view move, std::string_view comment) override {
        //std::cout << "moves started" << std::endl;
        std::string white_player = "";
        std::string black_player = "";
        std::string event, whiteElo, blackElo, timeControl, termination, site;
        std::string variant = "";

        for (const auto& [key, value] : headers) {
            if (key == "Event") event = value;
            if (key == "WhiteElo") whiteElo = value;
            if (key == "BlackElo") blackElo = value;
            if (key == "TimeControl") timeControl = value;
            if (key == "Termination") termination = value;
            if (key == "White") white_player = value;
            if (key == "Black") black_player = value;
            if (key == "Variant") variant = value;
            if (key == "Site") site = value;
            if (key == "Event") event = value;
        }

        

        std::string turn = getColor(clock_board);
        

        //std::cout << clock_board.getFen() << std::endl;

        std::string comment_str(comment);
        comment_str = trim(comment_str);

        //std::cout << comment_str << std::endl;
        //std::cout << move << std::endl;

        //std::regex clockRegex(R"(\[%clk\s+(\d+:\d+:\d+)\])");
        std::regex clockRegex(R"(\[%clk\s+(\d+:\d+:\d+(\.\d+)?)\])");
        std::smatch clockMatch;
        std::string clockTime;
        if (std::regex_search(comment_str, clockMatch, clockRegex)) {
            clockTime = clockMatch[1];  // Extract the time
        }

        if(clockTime==""){
            headers.clear();
            moves_with_turns.clear();
            comments.clear();
            white_clock_times.clear();
            black_clock_times.clear();
            clock_board = Board();
            //games_removed++;
            //total_games--;
            //games_processed--;
            return;
        }

        
        std::string fen;
        size_t index = comment_str.find(']');
        if (index != std::string::npos && index + 1 < comment_str.size()) {
            fen = trim(comment_str.substr(index + 1));  // Extract everything after ']'
        }

        
        //std::cout << "clock time:" << clockTime << "," << "fen string:" << fen << std::endl;
        if(fen==""){
            std::cout << "clock board fen:" << clock_board.getFen() << std::endl;
            std::cout << "move:" << move << std::endl;
            std::cout << "comment:" << comment << std::endl;
            exit(0);
        }

        std::string uci_move_str(move);
        uci_move_str.erase(std::remove_if(uci_move_str.begin(), uci_move_str.end(), [](char c) {
            return c == '+' || c == '#';
        }), uci_move_str.end());
        

        if(turn=="0"){
            white_clock_times.push_back(timeToSeconds(clockTime));
            white_move_count += 1;
            move_numbers.push_back(white_move_count - 1);
            white_moves.push_back(uci_move_str);
        }
        if(turn=="1"){
            black_clock_times.push_back(timeToSeconds(clockTime));
            black_move_count += 1;
            move_numbers.push_back(black_move_count - 1);
            black_moves.push_back(uci_move_str);
        }

        if (variant=="" && event!="Live Chess - Odds Chess"){
            Move uci_move = uci::uciToMove(clock_board, uci_move_str);
            //std::cout << white_player << std::endl;
            clock_board.makeMove(uci_move);
        }

        if (comment_str.find("eval") != std::string::npos) {
            
            size_t index = fen.find(']');
            if (index != std::string::npos && index + 1 < fen.size()) {
                fen = trim(fen.substr(index + 1));  // Extract everything after ']'
            }
        } 
        

        moves_with_turns.push_back(std::string(move)+":"+turn);
        fens.push_back(std::string(fen));
        comments.push_back(comment.empty() ? "" : std::string(comment));

        
    }

    void startPgn() override {
        
    }

    void startMoves() override {
        //std::cout << "start moves called" << std::endl;
    }

    void endPgn() override {
        
        int base_time = 0;
        int increment_time = 0;
        std::string white_player = "";
        std::string black_player = "";
        std::string event, whiteElo, blackElo, timeControl, termination;
        for (const auto& [key, value] : headers) {
            if (key == "Event") event = value;
            if (key == "WhiteElo") whiteElo = value;
            if (key == "BlackElo") blackElo = value;
            if (key == "TimeControl") timeControl = value;
            if (key == "Termination") termination = value;
            if (key == "White") white_player = value;
            if (key == "Black") black_player = value;
            if (key == "Variant"){
                games_removed++;
                headers.clear();
                moves_with_turns.clear();
                comments.clear();
                white_clock_times.clear();
                black_clock_times.clear();
                clock_board = Board();
                total_games--;
                return;
            }
            if (key == "Event"){
                if(value=="Live Chess - Odds Chess"){
                    games_removed++;
                    headers.clear();
                    moves_with_turns.clear();
                    comments.clear();
                    white_clock_times.clear();
                    black_clock_times.clear();
                    clock_board = Board();
                    total_games--;
                    return;
                }
            }
        }

        

        

        // Check conditions and print headers if any condition is met
        if (event.empty() || event == "Rated Correspondence game" ||
            whiteElo.empty() || whiteElo == "?" ||
            blackElo.empty() || blackElo == "?" ||
            timeControl.empty() || timeControl == "-" ||
            termination.empty() || termination == "Abandoned"){
            games_removed++;
            headers.clear();
            moves_with_turns.clear();
            comments.clear();
            white_clock_times.clear();
            black_clock_times.clear();
            clock_board = Board();
            total_games--;
            return;

            /*
            std::cout << "\nGame " << games_processed << " matched criteria:\n";
            for (const auto& [key, value] : headers) {
                std::cout << key << ": " << value << std::endl;
            }
            */
        }

        size_t split_pos = timeControl.find('+');
        if(split_pos == std::string::npos){
            base_time = std::stoi(timeControl);
            increment_time = 0;
        }
        else{
            base_time = std::stoi(timeControl.substr(0, split_pos));
            increment_time = std::stoi(timeControl.substr(split_pos + 1));
        }

        white_elapsed_times.push_back(0);
        black_elapsed_times.push_back(0);
        for(int i = 0; i<white_clock_times.size()-1; i++){
            white_elapsed_times.push_back(white_clock_times[i] - white_clock_times[i+1]);
        }
        for(int i = 0; i<black_clock_times.size()-1; i++){
            black_elapsed_times.push_back(black_clock_times[i] - black_clock_times[i+1]);
        }

        //printArray(white_moves);
        //printArray(white_elapsed_times);
        //printArray(black_moves);
        //printArray(black_elapsed_times);

        /*
            moves_with_times string format:
            if it's white's turn:
            fen_string:uci_move_str:turn:white_remaining_time:black_remaining_time:white_time_spent
            if it's black's turn:
            fen_string:uci_move_str:turn:white_remaining_time:black_remaining_time:black_time_spent
            turn is either "0" or "1", where "0" is for white and "1" is for black
        */

        int white_clock_iterator = 0;
        int black_clock_iterator = 0;
        fens.pop_back();
        for(int i = 0; i<=moves_with_turns.size()-1; i++){
            size_t pos = moves_with_turns[i].find(':');
            std::string move = moves_with_turns[i].substr(0, pos);
            std::string turn = moves_with_turns[i].substr(pos + 1);

            if(turn=="0"){
                if(black_clock_iterator==0){
                    moves_with_time.push_back(fens[i]+":"+move+":"+turn+":"
                    +std::to_string(white_clock_times[white_clock_iterator])+":"+
                    std::to_string(black_clock_times[black_clock_iterator])+":"+
                    std::to_string(white_elapsed_times[white_clock_iterator]));
                    white_clock_iterator++;
                }

                else{
                    moves_with_time.push_back(fens[i]+":"+move+":"+turn+":"
                    +std::to_string(white_clock_times[white_clock_iterator])+":"+
                    std::to_string(black_clock_times[black_clock_iterator-1])+":"+
                    std::to_string(white_elapsed_times[white_clock_iterator]));
                    white_clock_iterator++;
                }
            }
            else{
                moves_with_time.push_back(fens[i]+":"+move+":"+turn+":"
                +std::to_string(white_clock_times[white_clock_iterator-1])+":"
                +std::to_string(black_clock_times[black_clock_iterator])+":"+
                std::to_string(black_elapsed_times[black_clock_iterator]));
                black_clock_iterator++;
            }
            
        }

        //printArray(moves_with_time);

        //shuffle moves so model doesn't have bias
        std::random_device rd;  // Seed source
        std::mt19937 gen(rd()); // Mersenne Twister random generator
        std::shuffle(moves_with_time.begin(), moves_with_time.end(), gen);

        
        for(int i = 0; i<=moves_with_time.size()-1; i++){
            //std::string fen = fens[i];
            std::string line = moves_with_time[i];
            std::vector<std::string> parts = splitString(line, ':');
            std::string fen = parts[0];
            std::string move = parts[1];
            std::string turn = parts[2];

            //std::cout << fen << std::endl;

            //std::cout << line << std::endl;
            //std::cout << white_player << std::endl;
            //std::cout << parts[3] << std::endl;
            //printArray(parts);
            float white_remaining_time = std::stof(parts[3]);
            float black_remaining_time = std::stof(parts[4]);
            float time_spent = std::stof(parts[5]);
        
        
            //legal moves
            //std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
            Board board(fen);
            std::vector<std::string> legal_move_list = generateLegalMovesFromFEN(board);
            std::string side_to_move = getColor(board);
            std::string castling_rights = getCastlingRights(board);
            auto piece_stats = getPieceStats(board, side_to_move);

            int white_kingside_castling_rights = 0;
            int white_queenside_castling_rights = 0;
            int black_kingside_castling_rights = 0;
            int black_queenside_castling_rights = 0;
            std::vector<int> encoded_board = parse_board(fen);
            //std::cout << fen << std::endl;
            //printArray(encoded_board);

            //can white castle kingside?
            if (castling_rights.find('K') != std::string::npos) {
                white_kingside_castling_rights = 1;
            }else{
                white_kingside_castling_rights = 0;
            }

            //can white castle queenside?
            if (castling_rights.find('Q') != std::string::npos) {
                white_queenside_castling_rights = 1;
            }else{
                white_queenside_castling_rights = 0;
            }

            //can black castle kingside?
            if (castling_rights.find('k') != std::string::npos) {
                black_kingside_castling_rights = 1;
            }else{
                black_kingside_castling_rights = 0;
            }

            //can black castle queenside?
            if (castling_rights.find('q') != std::string::npos) {
                black_queenside_castling_rights = 1;
            }else{
                black_queenside_castling_rights = 0;
            }

            int from_square = SQUARES[move.substr(0, 2)];
            int to_square = SQUARES[move.substr(2, 2)];

            int result = 0;
            int white_rating = 0;
            int black_rating = 0;
            int total_full_moves = 0;
            for (const auto& [key, value] : headers) {
                if(trim(key) == "Result"){
                    if(trim(value)=="1-0"){
                        result = 1;
                    }
                    if(trim(value)=="1/2-1/2"){
                        result = 0;
                    }
                    if(trim(value)=="0-1"){
                        result = -1;
                    }
                }
                if(trim(key) == "WhiteElo"){
                    white_rating = std::stoi(trim(value));
                }
                if(trim(key) == "BlackElo"){
                    black_rating = std::stoi(trim(value));
                }
                if(trim(key) == "PlyCount"){
                    total_full_moves = static_cast<int>(std::ceil(std::stof(trim(value)) / 2.0));
                }
            }

            //std::cout << result << std::endl;

            int categorical_result = 0;
            if (result==1){
                categorical_result = 2;
            }
            if (result==0){
                categorical_result = 1;
            }
            if (result==-1){
                categorical_result = 0;
            }

            int phase = 0;
            if(piece_stats["total_pieces"] >= 26){
                phase = phase_encoder["opening"];
            }
            else if(piece_stats["total_pieces"] <= 26 && piece_stats["total_pieces"]>=14){
                phase = phase_encoder["middlegame"];
            }
            else{
                phase = phase_encoder["endgame"];
            }
            int white_material_value = piece_stats["white_material_value"];
            int black_material_value = piece_stats["black_material_value"];
            int material_difference = piece_stats["material_difference"];

            int move_number = board.fullMoveNumber();
            int moves_until_end = total_full_moves - move_number;

            if(time_spent == 0){
                std::random_device rd;  // Seed for the random number engine
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.1, 0.5);
                time_spent = dis(gen);
            }
            if(increment_time>0 || time_spent < 0){
                time_spent += increment_time;
            }
            
            if(turn == "0"){
                if(white_remaining_time == 0){
                    time_spent = 0.005;
                }
            }
            if(turn == "1"){
                if(black_remaining_time == 0){
                    time_spent = 0.005;
                }
            }



            

            //create dictionary
            Dictionary dict;
            dict.turn = std::stoi(side_to_move);
            dict.white_kingside_castling_rights = white_kingside_castling_rights;
            dict.white_queenside_castling_rights = white_queenside_castling_rights;
            dict.black_kingside_castling_rights = black_kingside_castling_rights;
            dict.black_queenside_castling_rights = black_queenside_castling_rights;
            for (int i = 0; i < 64; i++) {
                dict.board_position[i] = encoded_board[i];
            }
            dict.from_square = from_square;
            dict.to_square = to_square;
            dict.length = 10;
            dict.phase = phase;
            dict.result = result;
            dict.categorical_result = categorical_result;
            dict.base_time = base_time;
            dict.increment_time = increment_time;
            dict.white_remaining_time = white_remaining_time;
            dict.black_remaining_time = black_remaining_time;
            dict.white_rating = white_rating;
            dict.black_rating = black_rating;
            dict.time_spent_on_move = time_spent;
            dict.move_number = move_number;
            dict.num_legal_moves = legal_move_list.size();
            dict.white_material_value = white_material_value;
            dict.black_material_value = black_material_value;
            dict.material_difference = material_difference;
            dict.moves_until_end = moves_until_end;
            strncpy(dict.fen, fen.c_str(), sizeof(dict.fen) - 1);
            dict.fen[sizeof(dict.fen) - 1] = '\0';
            strncpy(dict.white_username, white_player.c_str(), sizeof(dict.white_username) - 1);
            dict.white_username[sizeof(dict.white_username) - 1] = '\0';
            strncpy(dict.black_username, black_player.c_str(), sizeof(dict.black_username) - 1);
            dict.black_username[sizeof(dict.black_username) - 1] = '\0';


            if(chunk.size() <= chunks_per_file-1){
                chunk.push_back(dict);
            }else{
                if(file_count == max_files_per_dir){
                    dir_count++;
                    std::filesystem::create_directory(output_dir+"/"+std::to_string(dir_count));
                    std::string out_file_path = output_dir + "/" + std::to_string(dir_count) + "/record_" + std::to_string(file_count) + ".zst";
                    write_to_file(chunk, out_file_path);
                    file_count = 0;
                    chunk.clear();
                    //file_count++;
                    chunk.push_back(dict);
                }else{
                    std::string out_file_path = output_dir + "/" + std::to_string(dir_count) + "/record_" + std::to_string(file_count) + ".zst";
                    write_to_file(chunk, out_file_path);
                    chunk.clear();
                    file_count++;
                    chunk.push_back(dict);
                }
            }
            
        }

        
        

        //printArray(fens);
        //printArray(moves);
        /*
        std::cout << fens.size() << std::endl;
        std::cout << moves_with_time.size() << std::endl;
        std::cout << white_elapsed_times.size() << std::endl;
        std::cout << black_elapsed_times.size() << std::endl;
        std::cout << white_clock_times.size() << std::endl;
        std::cout << black_clock_times.size() << std::endl;
        */
        

        headers.clear();
        moves_with_time.clear();
        moves_with_turns.clear();
        fens.clear();
        comments.clear();
        turns.clear();
        white_clock_times.clear();
        black_clock_times.clear();
        white_elapsed_times.clear();
        black_elapsed_times.clear();
        move_numbers.clear();
        white_move_count = 0;
        black_move_count = 0;
        clock_board = Board();
        fens.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        games_processed++;  // Increment game count

        

        // Calculate elapsed time
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = now - start_time;

        float iterations_per_second = static_cast<float>(games_processed) / elapsed_time.count();

        // Print updated progress on the same line
        float percent_complete = static_cast<float>(games_processed)*100/total_games;
        float predicted_remaining_time = static_cast<float>(total_games - games_processed) / iterations_per_second;

        std::cout << "\rGames processed: " << games_processed << " / " << total_games
                << " | " << percent_complete << "% | " << games_removed << " games removed | " <<
                 "Elapsed time: " << elapsed_time.count() << "s | " 
                 << iterations_per_second << "it/s | "
                 << "Time remaining: " << predicted_remaining_time << "s |" << std::flush;
                 
        //exit(0);
    }

};

// Function to count the number of games in the PGN file
int countGames(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return 0;

    int count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("[Event") != std::string::npos) {  // New game starts with [Event]
            count++;
        }
    }
    return count;
}

int main(int argc, char const* argv[]) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Check if it starts with "--"
        if (arg.rfind("--", 0) == 0) {
            size_t equal_pos = arg.find('=');
            if (equal_pos != std::string::npos) {
                std::string key = arg.substr(0, equal_pos);  // Extract "--option"
                std::string value = arg.substr(equal_pos + 1); // Extract "value"
                args[key] = value;
            } else {
                std::cerr << "Invalid argument format: " << arg << std::endl;
            }
        }
    }

    std::string file = "";
    int chunks_per_file = 0;
    int max_files_per_dir = 0;
    std::string output_dir = "";
    if (args.count("--pgn_file")) {
        file = args["--pgn_file"];

        size_t dot_pos = file.find_last_of('.');
        output_dir = file.substr(0, dot_pos);
    }else{
        std::cerr << "Must include --pgn_file arg, eg. --pgn_file=kasparov.pgn" << std::endl;
        exit(0);
    }

    if (args.count("--chunks_per_file")) {
        chunks_per_file = std::stoi(args["--chunks_per_file"]);
    }else{
        std::cerr << "Must include --chunks_per_file arg, eg. --chunks_per_file=1000" << std::endl;
        exit(0);
    }

    if (args.count("--max_files_per_dir")) {
        max_files_per_dir = std::stoi(args["--max_files_per_dir"]);
    }else{
        std::cerr << "Must include --max_files_per_dir arg, eg. --max_files_per_dir=100" << std::endl;
        exit(0);
    }

    if (args.count("--outputdir")) {
        output_dir = args["--outputdir"];
    }else{
        std::cerr << "--outputdir not specified, " << "will create new dir named " << output_dir << "." << std::endl;
    }


     
    std::ifstream file_stream(file);

    if (!file_stream) {
        std::cerr << "Error: Could not open PGN file: " << file << std::endl;
        return 1;
    }

    int total_games = countGames(file);
    if (total_games == 0) {
        std::cerr << "No games found in PGN file!" << std::endl;
        return 1;
    }

    int games_processed = 0;

    auto vis = std::make_unique<MyVisitor>(games_processed, total_games, output_dir, chunks_per_file, max_files_per_dir);
    
    pgn::StreamParser parser(file_stream);
    parser.readGames(*vis);

    std::cout << std::endl;  // Move to the next line after final output

    return 0;
}
