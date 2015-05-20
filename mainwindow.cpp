#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cudadevice.h"
#include "deviceinfowidget.h"
#include <QMessageBox>
#include <QStringBuilder>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    const int numDevices = CudaDevice::deviceCount();
    
    for (int i = 0; i < numDevices; ++i) {
        try {
            auto dw = new DeviceInfoWidget(i);
            
            ui -> tabWidget -> addTab(dw, "device &" % QString::number(i));
        } catch (CudaDevice::Exception e) {
            auto msgBox = new QMessageBox(this);
            
            msgBox -> setModal(false);
            msgBox -> setText(e.what());
            msgBox -> show();
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
