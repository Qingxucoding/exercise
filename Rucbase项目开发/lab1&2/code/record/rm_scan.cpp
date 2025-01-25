/* Copyright (c) 2023 Renmin University of China
RMDB is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
        http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

#include "rm_scan.h"

#include "rm_file_handle.h"

/**
 * @brief 初始化file_handle和rid
 * @param file_handle
 */

RmScan::RmScan(const RmFileHandle* file_handle) : file_handle_(file_handle) {
    // Todo:
    // 初始化file_handle和rid（指向第一个存放了记录的位置）
    // 初始化rid为第一个记录页面和未使用的插槽
    rid_ = { RM_FIRST_RECORD_PAGE, -1 };
    if (rid_.page_no < file_handle_->file_hdr_.num_pages) {
        next();  // 开始寻找第一个有效的记录
        return;
    }
    rid_.page_no = RM_NO_PAGE;  // 设置为无效状态
}

/**
 * @brief 找到文件中下一个存放了记录的位置
 */

void RmScan::next() {
    // Todo:
    // 目标：寻找文件中下一个存有记录的可用插槽，使用 rid_ 指向该位置
    const int max_records = file_handle_->file_hdr_.num_records_per_page;  // 每页最大记录数
    const int page_max = file_handle_->file_hdr_.num_pages;                // 总页数

    // 缓存当前页面的句柄
    RmPageHandle page_handle = file_handle_->fetch_page_handle(rid_.page_no);           // 获取当前页面句柄
    rid_.slot_no = Bitmap::next_bit(1, page_handle.bitmap, max_records, rid_.slot_no);  // 查找下一个有效插槽

    // 继续查找有效插槽，直到找到或超出页面范围
    while (rid_.slot_no == max_records) {
        file_handle_->buffer_pool_manager_->unpin_page(page_handle.page->get_page_id(), false);  // 解锁当前页面
        rid_.page_no++;  // 转到下一个页面
        if (rid_.page_no >= page_max) {
            return;  // 超过总页面数，返回
        }
        // 在这里缓存当前页面的句柄
        page_handle = file_handle_->fetch_page_handle(rid_.page_no);           // 获取下一个页面的句柄
        rid_.slot_no = Bitmap::first_bit(1, page_handle.bitmap, max_records);  // 查找该页面的第一个有效插槽
    }
}
/**
 * @brief 判断是否到达文件末尾
 */
bool RmScan::is_end() const {
    // Todo: 修改返回值
    // 如果当前页已经达到最大记录数，并且页数超出文件的总页数，则认为到达结尾
    bool is_slot_full = (rid_.slot_no == file_handle_->file_hdr_.num_records_per_page);
    bool is_page_out_of_bounds = (rid_.page_no >= file_handle_->file_hdr_.num_pages);
    return is_slot_full && is_page_out_of_bounds;
}
/**
 * @brief RmScan内部存放的rid
 */
Rid RmScan::rid() const { return rid_; }
