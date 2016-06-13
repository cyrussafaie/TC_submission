#define LOCAL

#include <bits/stdc++.h>
using namespace std;

#define cimg_use_png

#include <CImg.h>
#include "main.cpp"

using namespace cimg_library;

VS readFile(string fn, bool dropFirstRow = false) {
	const int MAX_LINE = 1000000;
	FILE *f = fopen(fn.c_str(), "r");
	VS rv;
	while (!feof(f)) {
		static char line[MAX_LINE];
		line[0] = 0;
		fgets(line, MAX_LINE, f);
		int n = strlen(line);
		while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
		if (n == 0) continue;
		if (dropFirstRow) {
			dropFirstRow = false;
			continue;
		}
		rv.PB(line);
	}
	rv.shrink_to_fit();
	return rv;
}

VS data;
VVS dataParts;
VC<VC<Sample>> dataSamples;

double SPLIT_SIZE = 0.66;
int RUNS = 10;

double runTest(int seed) {
	seed *= 1337;
	auto p = Utils::splitData(dataParts, SPLIT_SIZE, seed);
	
	VS trainData;
	VS testData;
	VS pred;
	map<int,string> truth[2];
	
	for (auto v : p.X) for (string &s : v) {
		trainData.PB(s);
	}
	
	for (auto v : p.Y) {
		Sample smp(v[0]);
		truth[smp.customerSegment][smp.productID] = classNames[smp.truth];
		for (string &s : v) {
			int pos = s.rfind(',');
			testData.PB(s.substr(0, pos));
		}
	}
	
	ElectronicPartsClassification algo;
	pred = algo.classifyParts(trainData, testData);
	
	double score = 0;
	int cnt = 0;
	REP(i, pred.S) {
		VS v = splt(pred[i], ',');
		int id = atoi(v[0].c_str());
		string s0 = v[1];
		string s1 = v[2];
		if (truth[0].count(id))	score += lossMatrix[classToID(s0)][classToID(truth[0][id])], cnt++;
		if (truth[1].count(id))	score += lossMatrix[classToID(s1)][classToID(truth[1][id])], cnt++;
	}
	return 1e6 * (1 - score / cnt);
}


const int IMG_RES_X = 1800;
const int IMG_RES_Y = 800;
const int MARGIN = 20;
const int RADIUS = 3;
void draw(VVPDD &data, VVPDD &bonus, string name, string fn) {
	double minx = 1e99;
	double maxx = -1e99;
	double maxy = 0;
	for (auto &v : data) for (auto &p : v) {
		minx = min(minx, p.X);
		maxx = max(maxx, p.X);
		maxy = max(maxy, p.Y);
	}
	for (auto &v : data) for (auto &p : v) {
		p.X = (p.X - minx) / (maxx - minx);
		p.Y /= maxy;
		p.X = MARGIN + p.X * (IMG_RES_X - 2 * MARGIN);
		p.Y = MARGIN + (1 - p.Y) * (IMG_RES_Y - 2 * MARGIN);
	}
	for (auto &v : bonus) for (auto &p : v) {
		p.X = (p.X - minx) / (maxx - minx);
		p.X = MARGIN + p.X * (IMG_RES_X - 2 * MARGIN);
		p.Y = MARGIN + (1 - p.Y) * (IMG_RES_Y - 2 * MARGIN);
	}
	
	for (auto &v : data) sort(ALL(v));
	for (auto &v : bonus) sort(ALL(v));
	
	unsigned char colors[][3] = {{255, 128, 128}, {128, 255, 128}, {128, 128, 255}};
	unsigned char white[3] = {255, 255, 255};
	CImg<unsigned char> img(IMG_RES_X, IMG_RES_Y, 1, 3);
	img = (unsigned char)0;
	REP(j, data.S) {
		auto &v = data[j];
		colors[1][0] = (BYTE)rng.next(50,255);
		colors[1][1] = (BYTE)rng.next(50,255);
		colors[1][2] = (BYTE)rng.next(50,255);
		REP(i, v.S) {
			img.draw_circle(v[i].X, v[i].Y, RADIUS, colors[min(1,j)]);
			if (i) img.draw_line(v[i-1].X, v[i-1].Y, v[i].X, v[i].Y, colors[min(1,j)]);
		}
	}
	REP(j, bonus.S) {
		auto &v = bonus[j];
		const unsigned char color[3] = {(BYTE)rng.next(50,255), (BYTE)rng.next(50,255), (BYTE)rng.next(50,255)};
		REP(i, v.S) {
			img.draw_circle(v[i].X, v[i].Y, RADIUS, color);
			if (i) img.draw_line(v[i-1].X, v[i-1].Y, v[i].X, v[i].Y, color);
		}
	}
	img.draw_line(MARGIN, IMG_RES_Y-MARGIN, IMG_RES_X-MARGIN, IMG_RES_Y-MARGIN, white);
	img.draw_line(MARGIN, IMG_RES_Y-MARGIN, MARGIN, MARGIN, white);
	
	img.draw_text(30, IMG_RES_Y-60, name.c_str(), white, NULL, 1, 23);
	img.save(fn.c_str());
}

void analyze() {
	auto groups = parseData(data);
	set<int> ids;
	map<int,int> values[2];
	for (auto group : groups) {
		values[group[0].customerSegment][group[0].productID] = group[0].truth;
		REP(i, group.S) {
			ids.insert(group[i].productID);
			assert(group[0].truth == group[i].truth);
		}
	}
	DB(groups.S);
	DB(ids.S);
	
	for (int id : ids) {
		cout << id << ' ' << (values[0].count(id)?classNames[values[0][id]]:"NA") << ' ' << (values[1].count(id)?classNames[values[1][id]]:"NA") << endl;
	}
	
	VVI cnt(2, VI(3));
	for (auto group : groups) {
		cnt[group[0].customerSegment][group[0].truth]++;
	}
	DB(cnt);
	
	VVS vvs;
	for (string s : data) vvs.PB(splt(s, ','));
	
	FOR(i, 1, vvs[0].S) {
		bool ok = true;
		map<string,string> mp;
		REP(j, vvs.S) {
			if (mp.count(vvs[j][0]) == 0) mp[vvs[j][0]] = vvs[j][i];
			if (mp[vvs[j][0]] != vvs[j][i]) ok = false;
		}
		if (ok) DB(i);
	}
	
	int sumCustomers = 0;
	int sumCustomersCnt = 0;
	int sumOrders = 0;
	int sumOrdersCnt = 0;
	for (auto group : groups) {
		auto customers = parseGroup(group);
		sumCustomers += customers.S;
		sumCustomersCnt++;
		for (auto customer : customers) {
			sumOrders += customer.S;
			sumOrdersCnt++;
		}
	}
	
	int count = 100;
	for (auto group : groups) {
		VD f = extractFeatures(group);
		cout << count; DB(f);
		
		auto customers = parseGroup(group);
		VVPDD graphs(1);
		VVPDD bonus;
		VPDD vv;
		VI date;
		for (Sample &s : group) {
			if (s.amount == 0) continue;
			graphs[0].emplace_back(s.date, s.cost / s.amount);
			vv.emplace_back(s.date, 0.05 + 0.05 * s.amount);
		}
		bonus.PB(vv);
		double y = 0.03;
		for (auto customer : customers) {
			VPDD v;
			for (Sample &s : customer) {
				date.PB(s.date);
				if (s.amount == 0) continue;
				v.emplace_back(s.date, s.sales / s.amount);
			}
			graphs.PB(v);
		}
		string s = i2s(group[0].productID) + ":" + i2s(group[0].customerSegment) + "[" + classNames[group[0].truth] + "] Days: " + i2s(Utils::calcMax(date)-Utils::calcMin(date));
		draw(graphs, bonus, s.c_str(), "graphs/x" + i2s(count++) + ".png");
	}
	
	DB(1.0 * sumCustomers / sumCustomersCnt);
	DB(1.0 * sumOrders / sumOrdersCnt);
	
	
	
}


int main(int argc, char **argv) {
	double startTime = getTime();
	
	data = readFile("example_data.csv", true);
	VI ids = getDataIDs(data);
	dataParts = VVS(*max_element(ALL(ids))+1);
	REP(i, data.S) dataParts[ids[i]].PB(data[i]);
	
	bool ANALYZE_MODE = false;
	
	FOR(i, 1, argc) {
		string cmd = argv[i];
		if (cmd == "-t") {
			THREADS_NO = atoi(argv[++i]);
		} else if (cmd == "-split") {
			SPLIT_SIZE = atof(argv[++i]);
		} else if (cmd == "-runs") {
			RUNS = atoi(argv[++i]);
		} else if (cmd == "-a") {
			ANALYZE_MODE = true;
		} else {
			cerr << "[Error] Unknown Command: " << cmd << endl;
			exit(1);
		}
	}
	
	if (ANALYZE_MODE) {
		analyze();
		exit(0);
	}
	
	double totalScore = 0;
	REP(seed, RUNS) {
		double score = runTest(seed);
		DB(score);
		totalScore += score; 
	}
	
	totalScore /= RUNS;
	cout << "Final Score: " << totalScore << endl;
	cout << "Time: " << (getTime() - startTime) << endl;
	
	return 0;
}
