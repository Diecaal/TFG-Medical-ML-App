#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <QMainWindow>
#include <QCheckBox>
#include <QPixmap>
#include <QPainter>
#include <QFileDialog>
#include <QMessageBox>
#include <QObject>
#include <QImage>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QHttpPart>
#include <QtNetwork/QHttpMultiPart>
#include <QtNetwork/QNetworkReply>
#include <QEventLoop>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <QGroupBox>

#include <QBarCategoryAxis>
#include <QChart>
#include <QBarSet>
#include <QHorizontalBarSeries>
#include <QValueAxis>
#include <QChartView>

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    QNetworkAccessManager *manager = new QNetworkAccessManager();
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void computeGraphics(QByteArray jsonDocument);

private slots:
    void on_buttonUpload_clicked();
    void on_btnUploadServer_clicked();

    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    QString selectedFilename;
    void generateDiseasesLabels();
    QString cleanJsonResponse(QByteArray responseBody);
};
#endif // MAINWINDOW_H
