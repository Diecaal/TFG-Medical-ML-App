#ifndef CRYPTOUTIL_H
#define CRYPTOUTIL_H

#include <iostream>
#include <iomanip>
#include <QDebug>
#include <vector>

using namespace std;

class CryptoUtil
{
public:
    CryptoUtil();
    static void encrypt(std::string plainText);
};

#endif // CRYPTOUTIL_H
