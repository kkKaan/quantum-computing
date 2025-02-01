#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using SatelliteId = int64_t;

void OnSatelliteReportedBack(SatelliteId satelliteId)
{
    // Don't change this code
    std::cout << "SatelliteReportedBack: " << satelliteId << std::endl;
}

void ErrDuplicateSatellite(SatelliteId satelliteId)
{
    // Don't change this code
    std::cout << "E1: " << satelliteId << std::endl;
}

void ErrInvalidSatellite(SatelliteId satelliteId)
{
    // Don't change this code
    std::cout << "E2: " << satelliteId << std::endl;
}

class SatelliteNetwork
{
public:
    void SatelliteConnected(SatelliteId satelliteId)
    {
        // Your code here...
        if (connected_satellites.count(satelliteId))
        {
            ErrDuplicateSatellite(satelliteId);
            return;
        }

        connected_satellites.insert(satelliteId);
        adjacency_map[satelliteId];
    }

    void RelationshipEstablished(SatelliteId satellite1, SatelliteId satellite2)
    {
        // Your code here...
        if (!connected_satellites.count(satellite1))
        {
            ErrInvalidSatellite(satellite1);
            return;
        }
        if (!connected_satellites.count(satellite2))
        {
            ErrInvalidSatellite(satellite2);
            return;
        }

        adjacency_map[satellite1].insert(satellite2);
        adjacency_map[satellite2].insert(satellite1);
    }

    void MessageReceived(const std::vector<SatelliteId>& notifiedSatellites)
    {
        // Your code here...
        // check notified satellites are connected
        for (auto id : notifiedSatellites)
        {
            if (!connected_satellites.count(id))
            {
                ErrInvalidSatellite(id);
                return;
            }
        }

        static const long NOT_NOTIFIED = -1;
        std::unordered_map<SatelliteId, long> notificationTimes;
        for (auto id : connected_satellites)
            notificationTimes[id] = NOT_NOTIFIED;

        for (auto id : notifiedSatellites)
            notificationTimes[id] = 0;

        std::unordered_map<SatelliteId, bool> processed;
        for (auto id : connected_satellites)
            processed[id] = false;

        std::vector<std::pair<long, SatelliteId>> finished;

        while (true)
        {
            long bestTime = LLONG_MAX;
            SatelliteId bestSat = -1;

            for (auto id : connected_satellites)
            {
                if (!processed[id] && notificationTimes[id] != NOT_NOTIFIED)   // notified but not processed
                {
                    if (notificationTimes[id] < bestTime || (notificationTimes[id] == bestTime && id < bestSat))
                    {
                        bestTime = notificationTimes[id];
                        bestSat = id;
                    }
                }
            }

            if (bestSat == -1)
                break;

            long curTime = bestTime;

            std::vector<SatelliteId> nbrs(adjacency_map[bestSat].begin(), adjacency_map[bestSat].end());
            sort(nbrs.begin(), nbrs.end());

            long finishFwd = curTime;
            for (auto nbr : nbrs)
            {
                finishFwd += 10;
                if (notificationTimes[nbr] == NOT_NOTIFIED || finishFwd < notificationTimes[nbr])
                {
                    notificationTimes[nbr] = finishFwd;
                }
            }

            long reportTime = finishFwd + 30;
            finished.push_back({ reportTime, bestSat });

            processed[bestSat] = true;
        }

        sort(finished.begin(), finished.end(),
             [](auto& a, auto& b)
             {
                 if (a.first != b.first)
                     return a.first < b.first;
                 return a.second < b.second;
             });

        for (auto& f : finished)
        {
            OnSatelliteReportedBack(f.second);
        }
    }
private:
    std::unordered_set<SatelliteId> connected_satellites;
    std::unordered_map<SatelliteId, std::unordered_set<SatelliteId>> adjacency_map;
};

int main()
{
    uint64_t N = 0u;
    std::cin >> N;

    SatelliteNetwork network;
    for (size_t i = 0; i < N; ++i)
    {
        std::string instructionText;
        std::cin >> instructionText;
        if (instructionText == "SatelliteConnected")
        {
            SatelliteId satelliteId = 0;
            std::cin >> satelliteId;
            network.SatelliteConnected(satelliteId);
        }
        else if (instructionText == "RelationshipEstablished")
        {
            SatelliteId fromSatelliteId = 0, toSatelliteId = 0;
            std::cin >> fromSatelliteId >> toSatelliteId;
            network.RelationshipEstablished(fromSatelliteId, toSatelliteId);
        }
        else if (instructionText == "MessageReceived")
        {
            uint64_t M = 0u;
            std::cin >> M;
            std::vector<SatelliteId> notifiedSatellites;

            for (int j = 0; j < M; ++j)
            {
                SatelliteId notifiedSatellite;
                std::cin >> notifiedSatellite;
                notifiedSatellites.push_back(notifiedSatellite);
            }
            network.MessageReceived(notifiedSatellites);
        }
        else
        {
            std::cerr << "Malformed input! " << instructionText << std::endl;
            return -1;
        }
    }

    return 0;
}