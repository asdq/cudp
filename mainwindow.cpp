#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cudadevice.h"
#include "deviceinfowidget.h"
#include <QStringBuilder>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    const int numDevices = CudaDevice::deviceCount();
    
    for (int i = 0; i < numDevices; ++i) {
        QString label = "device &" % QString::number(i);
        ui -> tabWidget -> addTab(new DeviceInfoWidget(i), label);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
