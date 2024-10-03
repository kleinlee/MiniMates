import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import AAAAA

Rectangle {
    width: 500
    height: 800
    color: "#000000FF"
    property string inConversationWith : "huahua"
    property int llm_load_status : -1

    QmlBridge {
        id: qmlBridge
        onLlmLoadedChanged:
        {
            if (qmlBridge.llmLoaded)
            {
                llm_load_status = 1
                loadButton.background.color = "#009900"
                loadButton.text = "Qwen2.5-4B-gguf"
            }
            else
            {
                llm_load_status = -1
                loadButton.background.color = "red"
                loadButton.text = "请重新加载"
                loadButton.enabled = true
            }
        }
    }

    // 上侧的栏
    Rectangle {
        id: topBar
        width: parent.width
        height: 45
        z: 1
        // 加载按钮
        Button {
            id: loadButton
            anchors.left: parent.left
            anchors.leftMargin: 20
            width: 170
            height: parent.height
//            z: 1
            text: "加载模型"
            onClicked: {
                loadButton.text = "加载中"
                loadButton.enabled = false
                var load_status = qmlBridge.loadLLMModel()
                console.log("SSSSSSSSS", load_status)
            }
        }
        ComboBox {
            id: comboBox_tts
            displayText: "音色: " + currentText
            model: [
                "晓晓（女）",
                "晓意（女）",
                "云间",
                "云希",
                "云霞",
                "云阳"]
            anchors.right: parent.right
            anchors.rightMargin: 20
            width: 170
            height: parent.height - 10
            anchors.verticalCenter: parent.verticalCenter
            onCurrentIndexChanged: {
                qmlBridge.setVoiceName(comboBox_tts.currentIndex);
                console.log("Selected item:", comboBox_tts.currentText, comboBox_tts.currentIndex)
            }
        }
    }


    Rectangle {
        id: rect0
        anchors.top: topBar.top  // 将 rect0 放在 topBar 下方
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom

        ColumnLayout {
            anchors.fill: parent

            ListView {
                id: listView
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: pane.leftPadding + messageField.leftPadding
                displayMarginBeginning: 40
                displayMarginEnd: 40
                verticalLayoutDirection: ListView.BottomToTop
                spacing: 12
                model: SqlConversationModel {
                    recipient: inConversationWith
                }
                delegate: Column {
                    anchors.right: sentByMe ? listView.contentItem.right : undefined
                    spacing: 6

                    readonly property bool sentByMe: model.recipient !== "Me"

                    Row {
                        id: messageRow
                        spacing: 6
                        anchors.right: sentByMe ? parent.right : undefined

                        Image {
                            id: avatar
                            width: 50
                            height: 50
                            source: !sentByMe ? "qrc:/resource/images/" + model.author.replace(" ", "_") + ".jpg" : ""
                        }

                        Rectangle {
                            width: Math.min(messageText.implicitWidth + 24, listView.width - avatar.width - messageRow.spacing)
                            height: messageText.implicitHeight + 24
                            radius: 10
                            color: sentByMe ? "lightgrey" : "steelblue"

                            TextEdit  {
                                id: messageText
                                text: model.message
                                font.pixelSize: 14
                                color: sentByMe ? "black" : "white"
                                anchors.fill: parent
                                anchors.margins: 12
                                wrapMode: Text.WordWrap
                                selectByMouse: true
                                readOnly: true
                            }
                        }
                    }

                    Label {
                        id: timestampText
                        text: Qt.formatDateTime(model.timestamp, "d MMM hh:mm")
                        color: "lightgrey"
                        anchors.right: sentByMe ? parent.right : undefined
                    }
                }

                ScrollBar.vertical: ScrollBar {}
            }

            Pane {
                id: pane
                Layout.fillWidth: true

                RowLayout {
                    width: parent.width

                    TextArea {
                        id: messageField
                        Layout.fillWidth: true
                        placeholderText: qsTr("Compose message")
                        wrapMode: TextArea.Wrap
                        Keys.onPressed: (event)=> {
                            if (event.key === Qt.Key_Return) {
                                verifyInput();
                            }
                        }
                        onTextChanged: {
                            if (length > 30) remove(30, length);
                        }
                    }

                    Button {
                        id: sendButton
                        text: listView.model.chatFinished ? qsTr("发送"):qsTr("停止")
                        enabled: !listView.model.chatFinished | messageField.length > 0
                        onClicked: {
                            // 点击开始，llm开始运行，按钮变为“停止”。llm运行结束
                            // 先判断当前llm对话是否进行中，若进行中按钮要显示"停止"
                            if (!listView.model.chatFinished)
                            {
                                listView.model.abort();
                            }
                            else
                            {
                                verifyInput();
                            }
                        }
                    }
                    Button {
                        id: clearButton
                        text: qsTr("清空历史")
                        onClicked: {
                            listView.model.removeAllMessage();
                        }
                    }
                }
            }
        }
    }
    // 自定义函数 func0
    function verifyInput() {
        if (llm_load_status < 1)
        {
            loadButton.background.color = "red"
            return;
        }

        var textWithoutSpaces = messageField.text.replace(/\s/g, '');
        if (textWithoutSpaces === "") {
            console.log("TextArea 只包含空白字符或换行符");
            messageField.text = "";
            return;
        } else {
            console.log("TextArea 包含实际字符");
        }
        if(listView.model.chatFinished & messageField.length > 0)
        {
            listView.model.sendMessage(inConversationWith, messageField.text);
            messageField.text = "";
        }
    }
}

