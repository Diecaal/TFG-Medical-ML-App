#ifndef LOGIN_H
#define LOGIN_H

#include <QDialog>
#include <QMessageBox>
#include <iostream>
#include <QDebug>

#include <QMouseEvent>
#include <QPoint>

#include <mainwindow.h>
#include <cryptoUtil.h>
#include <regex>

namespace Ui {
class LogIn;
}

class LogIn : public QWidget {
    Q_OBJECT

public:
    explicit LogIn(QWidget *parent = nullptr);
    ~LogIn();

private slots:

    void on_pushButton_login_clicked();

    void on_pushButton_hopistal_domain_clicked();

private:
    Ui::LogIn *ui;

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;

    QPoint cur_pos;
    QPoint new_pos;
};

#endif // LOGIN_H
