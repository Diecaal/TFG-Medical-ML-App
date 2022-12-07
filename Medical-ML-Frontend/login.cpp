#include "login.h"
#include "ui_login.h"

LogIn::LogIn(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LogIn)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::WindowType::FramelessWindowHint);
    this->setAttribute(Qt::WA_TranslucentBackground);

    this->getServerStatus();
}

LogIn::~LogIn()
{
    delete ui;
}

void LogIn::on_pushButton_login_clicked()
{
    QString email = ui->lineEdit_email->text();
    QString password = ui->lineEdit_password->text();

    if(email.trimmed().isEmpty()) {
        QMessageBox::warning(this, "Log In", "Enter your email");
        return;
    }

    if(password.trimmed().isEmpty()) {
        QMessageBox::warning(this, "Log In", "Enter your password");
        return;
    }

    const std::regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");
    if(!(std::regex_match(email.toLocal8Bit().constData(), pattern))) {
        QMessageBox::warning(this, "Log In", "Email format is not valid");
        return;
    }

    CryptoUtil::encrypt(password.toStdString());

    if(email == "test@hospital.gov" && password == "test") {
        this->hide();
        MainWindow *mainWindow = new MainWindow();
        mainWindow->show();
    } else {
        QMessageBox::critical(this, "Log In", "Email or password are incorrect");
    }
}


void LogIn::on_pushButton_hopistal_domain_clicked()
{
    QString formated = ui->lineEdit_email->text();
    formated.append(QString("test@hospital.gov"));

    ui->lineEdit_password->text().append("test");

    ui->lineEdit_email->setText(formated);
}

void LogIn::mousePressEvent(QMouseEvent *event) {
    cur_pos = event->globalPosition().toPoint();
}

void LogIn::mouseMoveEvent(QMouseEvent *event) {
    new_pos = QPoint(event->globalPosition().toPoint() - cur_pos);
    move(x() + new_pos.x(), y() + new_pos.y());
    cur_pos = event->globalPosition().toPoint();
}

string LogIn::stringToSHA(string password)
{

}

void LogIn::getServerStatus()
{
    ui->labelStatusImg->setPixmap(QPixmap(":/media/media/checked.png"));
    ui->labelStatusImg->setScaledContents(true);
    ui->labelStatusImg->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
}

void LogIn::on_pushButton_clicked()
{
    QApplication::quit();
}

