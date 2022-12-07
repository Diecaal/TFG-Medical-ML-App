#include "mainwindow.h"
#include "ui_mainwindow.h"

#include<QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->generateDiseasesLabels();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_buttonUpload_clicked()
{
    selectedFilename = QFileDialog::getOpenFileName(this, "Upload Image", "C:/", tr("Image Files (*.png *.jpg *.bmp)")); //, 0, QFileDialog::DontUseNativeDialog);

    if(QString::compare(selectedFilename, QString()) != 0)
    {
        QImage image;
        bool valid = image.load(selectedFilename);

        if(valid)
        {
            image = image.scaledToWidth(ui->labelUpload->width(), Qt::SmoothTransformation);

            QPixmap pixImage = QPixmap::fromImage(image);

            int w = ui->labelUpload->width();
            int h = ui->labelUpload->height();

            ui->labelUpload->setPixmap(pixImage.scaled(w,h,Qt::KeepAspectRatio));
            QMessageBox::information(this, "Image uploaded", selectedFilename);

            ui->btnUploadServer->setEnabled(true);
        }
        else
        {
            QMessageBox::warning(this, "Image could not be selected", selectedFilename);
            ui->btnUploadServer->setEnabled(false);
        }
    }
    else
    {
        QMessageBox::warning(this, "No image selected", selectedFilename);
        ui->btnUploadServer->setEnabled(false);
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

    actionDiseasesButtons(true);
    ui->btnGenerateGradcam->setEnabled(true);
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
    QBarSet *barSet = new QBarSet("Prediction");

    currentImgUUID = jsonObj["uuid"].toString();
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

    ui->btnDiseasesLayout->setAlignment(Qt::AlignCenter);
    ui->btnDiseasesLayout->setSpacing(20);
    actionDiseasesButtons(false);
}

void MainWindow::actionDiseasesButtons(bool enable)
{
    QList<QCheckBox*> allButtons = ui->btnDiseasesLayout->parentWidget()->findChildren<QCheckBox*>();
    for(int i = 0; i < allButtons.size(); i++) {
        allButtons.at(i)->setEnabled(enable);
    }

}


void MainWindow::on_btnGenerateGradcam_clicked()
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

    QMainWindow* window = new QMainWindow();
    QWidget* widget = new QWidget();
    QGridLayout* grid = new QGridLayout();

    int row = 0;
    int col = 0;
    for(int i = 0; i < selectedDiseases.size(); i++) {
        QNetworkRequest request;
        request.setUrl(QUrl("http://127.0.0.1:80/gradcam/" + currentImgUUID + "/" + selectedDiseases.at(i)));
        qInfo() << "http://127.0.0.1:80/gradcam/" + currentImgUUID + "/" + selectedDiseases.at(i);
        QNetworkReply* reply = manager->get(request);

        QEventLoop loop;
        connect(reply, SIGNAL(finished()),&loop,SLOT(quit()));
        loop.exec();

        QPixmap pm;
        pm.loadFromData(reply->readAll());
        pm.scaled(528,528,Qt::KeepAspectRatio);

        QLabel *imageLabel = new QLabel();
        imageLabel->setPixmap(pm);
        if(col > 1){
            col = 0;
            row ++;
        }
        grid->addWidget(imageLabel,row,col,1,1);
        col++;
    }

    window->setWindowTitle("Gradcam of XRay for selected diseases");
    widget->setLayout(grid);
    window->setCentralWidget(widget);
    window->resize(1000, 500);
    window->show();

    for(int i = 0; i < allButtons.size(); i++) {
        allButtons.at(i)->setDisabled(false);
    }
}

