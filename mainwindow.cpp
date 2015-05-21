#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "deviceinfowidget.h"
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    addDevices();
    initHelp();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onActionAbout()
{
    m_messageBox -> exec();
}

/*
 * @brief Create a tab for each device.
 */
void MainWindow::addDevices()
{
    const int numDevices = CudaDevice::deviceCount();
    
    for (int i = 0; i < numDevices; ++i) {
        try {
            CudaDevice dev(i);
            auto dw = new DeviceInfoWidget;
            auto label = QString("Device &%1").arg(i);
            
            dw -> setData(dev);
            ui -> tabWidget -> addTab(dw, label);
        } catch (CudaDevice::Exception e) {
            auto msgBox = new QMessageBox(this);
            
            msgBox -> setModal(false);
            msgBox -> setText(e.what());
            msgBox -> show();
        }
    }
}

/*
 * @brief Initialize Help dialogs.
 */
void MainWindow::initHelp()
{
    QString about = tr("<p align=\"center\">"
                       "<b>Display CUDA devices v0.2</b></p>"
                       "<p align=\"justify\">Lookup for CUDA devices. "
                       "For each device found, display its properties.</p>"
                       "<p align=\"right\"><b>Author: Fabio Vaccari</b><p>");
    
    m_messageBox = new QMessageBox(this);
    m_messageBox -> setTextFormat(Qt::RichText);
    m_messageBox -> setWindowTitle(tr("About cudp"));
    m_messageBox -> setText(about);
    
    connect(ui -> actionAbout, SIGNAL(triggered()), SLOT(onActionAbout()));
    connect(ui -> actionAboutQt, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}
