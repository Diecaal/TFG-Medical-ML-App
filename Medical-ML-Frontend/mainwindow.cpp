#include "mainwindow.h"
#include "ui_mainwindow.h"

#include<QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QPixmap pix(":/media/media/background.jpg");
    pix = pix.scaled(this->size(), Qt::IgnoreAspectRatio);

    QPalette palette;
    palette.setBrush(QPalette::Window, pix);
    this->setPalette(palette);

    this->generateDiseasesLabels();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_buttonUpload_clicked()
{
    selectedFilename = QFileDialog::getOpenFileName(this, "Upload Image", QDir::home().absolutePath(), tr("Image Files (*.png *.jpg *.bmp)"));

    if(QString::compare(selectedFilename, QString()) != 0)
    {
        QImage image;
        bool valid = image.load(selectedFilename);

        if(valid)
        {
            image = image.scaledToWidth(ui->labelUpload->width(), Qt::SmoothTransformation);

            ui->labelUpload->setPixmap(QPixmap::fromImage(image));
            QMessageBox::information(this, "Image uploaded", selectedFilename);

            ui->btnUploadServer->setEnabled(true);
        }
        else
        {
            QMessageBox::warning(this, "Image could not be selected", selectedFilename);
            ui->btnUploadServer->setEnabled(false);
        }
    }
}


void MainWindow::on_btnUploadServer_clicked()
{
    QNetworkRequest request;
    request.setUrl(QUrl("http://127.0.0.1:80/predict"));

    QHttpMultiPart* multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QFile* file = new QFile(selectedFilename);
    file->open(QIODevice::ReadOnly);

    // Takes out the path from the file name
    QFileInfo fileInfo(file->fileName());
    QString fileName = fileInfo.fileName();

    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/png"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"file\"; filename=\"" + fileName + "\""));

    imagePart.setBodyDevice(file);
    file->setParent(multiPart); // we cannot delete the file now, so delete it with the multiPart

    multiPart->append(imagePart);

    ui->lineEditStatusUpload->setText(QString("Computing predictions in server..."));
    QNetworkReply* reply = manager->post(request, multiPart);

    // Waits for POST method to be finished and returns its body data
    QEventLoop loop;
    connect(reply, SIGNAL(finished()),&loop,SLOT(quit()));
    loop.exec();
    // Ahead from here we have access to reply data

    computeGraphics(reply->readAll());
}

QString MainWindow::cleanJsonResponse(QByteArray responseBody) {
    QString resp = responseBody.toStdString().c_str();
    QRegularExpression backSlashes("\\\\+");
    resp.replace(backSlashes,"");
    resp = resp.mid(1,resp.length()-2);
    resp = resp.simplified();
    resp.replace(" ","");

    return resp;
}

void MainWindow::computeGraphics(QByteArray responseBody) {

    QString resp = this->cleanJsonResponse(responseBody);

    QJsonDocument jsonDoc = QJsonDocument::fromJson(resp.toUtf8());
    QJsonObject jsonObj;

    // check validity of the document
    if(!jsonDoc.isNull())
    {
        if(jsonDoc.isObject())
        {
            jsonObj = jsonDoc.object();
        }
        else
        {
            qDebug() << "Document is not an object";
        }
    }
    else
    {
        qDebug() << "Invalid JSON...\n";
    }

    QStringList categories;
    QHorizontalBarSeries *series = new QHorizontalBarSeries();
    QBarSet *barSet = new QBarSet("Prediccion");

    QJsonArray predictionsArray = jsonObj["predictions"].toArray();

    for(int i = 0; i < predictionsArray.size(); i++) {
        barSet->append( predictionsArray.at(i).toObject().value("prediction").toDouble() );
        categories.append( predictionsArray.at(i).toObject().value("disease").toString() );
    }

    series->append( barSet );

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Prediction per disease");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    QBarCategoryAxis *axisY = new QBarCategoryAxis();
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    QValueAxis *axisX = new QValueAxis();
    axisX->setRange(0,1);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    //axisX->applyNiceNumbers();

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *view = new QChartView(chart);
    view->setRenderHint(QPainter::Antialiasing);


    QMainWindow* window = new QMainWindow();
    window->setWindowTitle("Predictions for given XRay");
    window->setCentralWidget(view);
    window->resize(1500, 570);
    window->show();
}

void MainWindow::generateDiseasesLabels() {
    QNetworkRequest request;
    request.setUrl(QUrl("http://127.0.0.1:80/diseases"));

    QNetworkReply* reply = manager->get(request);

    QEventLoop loop;
    connect(reply, SIGNAL(finished()),&loop,SLOT(quit()));
    loop.exec();

    QString response = this->cleanJsonResponse(reply->readAll());
    qInfo() << response;

    QJsonDocument jsonDoc = QJsonDocument::fromJson(response.toUtf8());
    QJsonObject jsonObj;

    // check validity of the document
    if(!jsonDoc.isNull())
    {
        if(jsonDoc.isObject())
        {
            jsonObj = jsonDoc.object();
        }
        else
        {
            qDebug() << "Document is not an object";
        }
    }
    else
    {
        qDebug() << "Invalid JSON...\n";
    }

    QJsonArray diseasesArray = jsonObj["diseases"].toArray();

    //int numDiseases = diseasesArray.size();
    int numColumns = 2;

    int currentRow = 0;
    int currentColumn = 0;
    foreach(const QJsonValue & disease, diseasesArray) {
        QCheckBox *cb = new QCheckBox(disease.toString(), this);
        QFont font = cb->font();
        font.setBold(true);
        font.setPointSize(12);
        cb->setFont(font);
        cb->setStyleSheet("color: white");
        if(currentColumn > (numColumns - 1)) {
            currentColumn = 0;
            ++currentRow;
        }
        ui->btnDiseasesLayout->addWidget(cb,currentRow,currentColumn,1,1);
        currentColumn++;
    }
}

void MainWindow::on_pushButton_clicked()
{
    QList<QCheckBox*> allButtons = ui->btnDiseasesLayout->parentWidget()->findChildren<QCheckBox*>();
    QList<QString> selectedDiseases;

    for(int i = 0; i < allButtons.size(); i++) {
        if(allButtons.at(i)->isChecked()) {
            selectedDiseases.append(allButtons.at(i)->text());
        }
        allButtons.at(i)->setDisabled(true);
        allButtons.at(i)->setChecked(false);
    }

    for(int i = 0; i < selectedDiseases.size(); i++) {
        qInfo() << selectedDiseases.at(i);
    }
}

