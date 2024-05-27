// pages/mine/mine.js
Page({
    data: {
        items: [
            {
                icon:'icon_cell_01.png',
                text:'邮件',
                mark:'开启美好一天',
                count: 10,
                subtitle: '封'
            },
            {
                icon:'icon_cell_02.png',
                text:'未读邮件',
                mark:'购买VIP\n直击效率模式',
                count: 10,
                subtitle: '未读'
            },
            {
                icon:'icon_cell_03.png',
                text:'垃圾邮件',
                mark:'购买SVIP\n开启垃圾邮件统计',
                count: 10,
                subtitle: '封'
            }
        ],
    },

    loginTap(){
        wx.navigateTo({
          url: '/pages/login/login',
        })
    },

    openDetailPage(event) {
        const index = event.currentTarget.dataset.index
        console.log('当前点击index:' + index);
    }
})