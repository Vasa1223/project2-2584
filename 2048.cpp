/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * 2048.cpp: Main file for the 2048 framework
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 * 
 * 
 * 11/5
 * 500000  avg = 217453, max = 559420, ops = 1886675 (1399046|5771957)        610     100%    (0.1%)
 *      987     99.9%   (0.3%)
 *      1597    99.6%   (1.8%)
 *      2584    97.8%   (1.4%)
 *      4181    96.4%   (5.8%)
 *      6765    90.6%   (19.8%)
 *      10946   70.8%   (48.1%)
 *      17711   22.7%   (21.8%)
 *      28657   0.9%    (0.9%)
 *
 * 
 * command:
 * .\2048.exe --total=100000 --block=1000 --limit=1000 --play="init alpha=0.0125"
 * .\2048.exe --total=500000 --block=1000 --limit=1000 --play="init alpha=0.00625"
 * .\2048.exe --total=100000 --block=1000 --limit=1000 --play="init alpha=0.00625"
 * .\2048.exe --total=150000 --block=1000 --limit=1000 --play="init alpha=0.0125 load=16x5.bin save=16x5.bin"
 */

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "board.h"
#include "action.h"
#include "agent.h"
#include "episode.h"
#include "statistic.h"

int main(int argc, const char* argv[]) {
	std::cout << "2584-Demo: ";
	std::copy(argv, argv + argc, std::ostream_iterator<const char*>(std::cout, " "));
	std::cout << std::endl << std::endl;

	size_t total = 1000, block = 0, limit = 0;
	std::string play_args, evil_args;
	std::string load, save;
	bool summary = false;
	for (int i = 1; i < argc; i++) {
		std::string para(argv[i]);
		if (para.find("--total=") == 0) {
			total = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--block=") == 0) {
			block = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--limit=") == 0) {
			limit = std::stoull(para.substr(para.find("=") + 1));
		} else if (para.find("--play=") == 0) {
			play_args = para.substr(para.find("=") + 1);
		} else if (para.find("--evil=") == 0) {
			evil_args = para.substr(para.find("=") + 1);
		} else if (para.find("--load=") == 0) {
			load = para.substr(para.find("=") + 1);
		} else if (para.find("--save=") == 0) {
			save = para.substr(para.find("=") + 1);
		} else if (para.find("--summary") == 0) {
			summary = true;
		}
	}

	statistic stat(total, block, limit);

	if (load.size()) {
		std::ifstream in(load, std::ios::in);
		in >> stat;
		in.close();
		summary |= stat.is_finished();
	}

	player play(play_args);
	rndenv evil(evil_args);

	while (!stat.is_finished()) {
		play.open_episode("~:" + evil.name());
		evil.open_episode(play.name() + ":~");

		stat.open_episode(play.name() + ":" + evil.name());
		episode& game = stat.back();
		while (true) {
			agent& who = game.take_turns(play, evil);
			action move = who.take_action(game.state());
			if (game.apply_action(move) != true) break;
			if (who.check_for_win(game.state())) break;
		}
		agent& win = game.last_turns(play, evil);
		stat.close_episode(win.name());

		play.close_episode(win.name());
		evil.close_episode(win.name());
	}

	if (summary) {
		stat.summary();
	}

	if (save.size()) {
		std::ofstream out(save, std::ios::out | std::ios::trunc);
		out << stat;
		out.close();
	}

	return 0;
}
