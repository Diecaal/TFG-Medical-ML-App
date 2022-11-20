#include "cryptoUtil.h"

CryptoUtil::CryptoUtil()
{

}

void CryptoUtil::encrypt(std::string plainText)
{
    vector<unsigned char> ciphered;
    vector<unsigned char> key = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };

    AES aes(AESKeyLength::AES_256);

    qDebug() << vector<unsigned char>(plainText.begin(), plainText.end()).size();
    ciphered = aes.EncryptECB(vector<unsigned char>(plainText.begin(), plainText.end()), key);

    vector<unsigned char> unciphered;

    unciphered = aes.DecryptECB(ciphered, key);

    qDebug() << QString::fromStdString(plainText);
    qDebug() << QString::fromStdString(string(ciphered.begin(), ciphered.end()));
    qDebug() << QString::fromStdString(string(unciphered.begin(), unciphered.end()));
}
