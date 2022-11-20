#ifndef LOGIN_H
#define LOGIN_H

#include <QDialog>
#include <QMessageBox>
#include <iostream>
#include <QDebug>

#include <mainwindow.h>
#include <cryptoUtil.h>
#include <regex>



namespace Ui {
class LogIn;
}

class LogIn : public QDialog
{
    Q_OBJECT

public:
    explicit LogIn(QWidget *parent = nullptr);
    ~LogIn();

private slots:
    void on_pushButton_login_clicked();

    void on_pushButton_hopistal_domain_clicked();

private:
    Ui::LogIn *ui;
};

#endif // LOGIN_H
