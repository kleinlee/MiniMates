#include "qmlbridge.h"
#include <QDateTime>
#include <QProcess>
#include <QDebug>
#include <QTcpServer>
#include <QEventLoop>
bool isPortInUse(quint16 port) {
    QTcpServer server;
    if (server.listen(QHostAddress::Any, port)) {
        server.close();
        return false; // 端口未被占用
    } else {
        return true; // 端口被占用
    }
}
QmlBridge::QmlBridge()
{
    human_assets_ = HumanAssets::GetInstance();
}
void QmlBridge::setVoiceName(int index)
{
    QString voice_name;
    switch (index)
    {
    case 0:
        voice_name = "zh-CN, XiaoxiaoNeural";
        break;
    case 1:
        voice_name = "zh-CN, XiaoyiNeural";
        break;
    case 2:
        voice_name = "zh-CN, YunjianNeural";
        break;
    case 3:
        voice_name = "zh-CN, YunxiNeural";
        break;
    case 4:
        voice_name = "zh-CN, YunxiaNeural";
        break;
    case 5:
        voice_name = "zh-CN, YunyangNeural";
        break;
    default:
        break;
    }
    human_assets_->m_TTS_client.setVoice(voice_name);
}
int QmlBridge::loadLLMModel()
{
    QString executablePath = "llama-server.exe";
    if (!QFile::exists(executablePath)) {
        qDebug() << "Executable not found: " << executablePath;
        return -1;
    }

    quint16 port = 8080;
    if (isPortInUse(port)) {
        qDebug() << "Port 8080 is in use. Trying another port...";
        port = 8081;
        while (isPortInUse(port)) {
            port++;
            if (port > 8090) { // 设置一个上限
                qDebug() << "No available port found.";
                return -1;
            }
        }
    }
    human_assets_->m_port_num = port;
    qDebug() << "Using port:" << port;

    // 使用QProcess运行指令
    QStringList arguments;
    arguments << "-m" << "Qwen.gguf" << "--port" << QString::number(port);
    m_llm_process.start(executablePath, arguments);

    if (!m_llm_process.waitForStarted(10000)) {
        qDebug() << "Failed to start process." << m_llm_process.errorString();
        return -1;
    }

    // 连接信号以处理进程输出
    connect(&m_llm_process, &QProcess::readyReadStandardOutput, this, &QmlBridge::handleProcessOutput);
    connect(&m_llm_process, &QProcess::readyReadStandardError, this, &QmlBridge::handleProcessOutput);

    if (m_llm_process.state() == QProcess::Running) {
        return 1;
    } else {
        return -1;
    }
}

void QmlBridge::handleProcessOutput()
{
    QByteArray output = m_llm_process.readAllStandardOutput();
    QByteArray errorOutput = m_llm_process.readAllStandardError();

    qDebug() << "Standard Output:" << output;
    qDebug() << "Standard Error:" << errorOutput;

    // 根据输出内容判断进程是否成功启动
    if (output.contains("starting the main loop")) {
        m_llm_loaded = true;
    } else {
        m_llm_loaded = false;
    }
    emit llmLoadedChanged();
}
