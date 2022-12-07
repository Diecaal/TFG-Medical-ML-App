#include "cryptoUtil.h"
#include "hashing/sha256.h"

CryptoUtil::CryptoUtil()
{

}

void CryptoUtil::encrypt(std::string plainText)
{
    SHA256 sha256;

    qDebug() << "hashed" << QString::fromStdString(sha256(plainText));
}
