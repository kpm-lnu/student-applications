#ifndef MY_SERVOS_H
#define MY_SERVOS_H

#include "defs.h"
#include "freertos/queue.h"


// #define SERVO_PLUG0_CH  0   // if calls 0 chan no matter which pin is under it, mutexes start race condition / deadlock
// #define SERVO_PLUG1_CH  1   // if calls 0 chan no matter which pin is under it, mutexes start race condition / deadlock
#define SERVO_PAN_CH    LEDC_CHANNEL_1   // to pin 14
#define SERVO_TILT_CH   LEDC_CHANNEL_2   // to pin 15

#define SERVO_PAN_ANGLE_BOTTOM      0
#define SERVO_PAN_ANGLE_TOP         180
#define SERVO_TILT_ANGLE_BOTTOM     0
#define SERVO_TILT_ANGLE_TOP        180

extern bool are_servos_inited;

#define SERVO_QUEUE_LEN     5
extern QueueHandle_t servo_queue;


/**
 * @brief   Seervos managing implemented via queue only with servo task
 * 
 * 
 */

esp_err_t init_my_servos();
esp_err_t deinit_my_servos();

esp_err_t run_servo_manager();


#endif