#include <iostream>
#include <fstream>
#include <array>
#include <vector>

class Point {
    public:
        std::vector<int> row;
        int classification=0;
        int getTrue();
    protected:
        int true_classification;

};

class KNearest {
    public:
        void setK();
        std::vector<Point> df;
        std::vector<Point> test_df;
        
    protected:
        int k;

    public:
        void setTrainData(int size, std::vector<std::vector<int>> data);
        void preprocess();
        void classifyTrain(int k);
        void testData(int test_size, std::vector<std::vector<int>> test_data);

};