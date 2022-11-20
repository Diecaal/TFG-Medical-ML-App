#include "login.h"
#include "ui_login.h"

LogIn::LogIn(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::LogIn)
{
    ui->setupUi(this);
}

LogIn::~LogIn()
{
    delete ui;
}

void LogIn::on_pushButton_login_clicked()
{
    QString email = ui->lineEdit_email->text();
    QString password = ui->lineEdit_password->text();

    const std::regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");
    if(!(std::regex_match(email.toLocal8Bit().constData(), pattern))) {
        QMessageBox::information(this, "Log In", "Email format is not valid");
        return;
    }

    if(email == "test@hospital.gov" && password == "test") {
        QMessageBox::information(this, "Log In", "Email and password is correct");
        this->hide();
        MainWindow *mainWindow = new MainWindow();
        mainWindow->show();
    } else {
        QMessageBox::information(this, "Log In", "Email or password are incorrect");
    }
}


void LogIn::on_pushButton_hopistal_domain_clicked()
{
    QString formated = ui->lineEdit_email->text();
    formated.append(QString("test@hospital.gov"));

    ui->lineEdit_password->text().append("test");

    ui->lineEdit_email->setText(formated);
}

