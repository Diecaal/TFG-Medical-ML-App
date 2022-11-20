#ifndef CRYPTOUTIL_H
#define CRYPTOUTIL_H

#include <iostream>
#include <iomanip>
#include <QDebug>
#include <vector>

#include <aes.h>

using namespace std;

class CryptoUtil
{
public:
    CryptoUtil();
    void encrypt(std::string plainText);
};

#endif // CRYPTOUTIL_H
