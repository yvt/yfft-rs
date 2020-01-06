extern crate yfft;

use std::rc::Rc;
use yfft::*;

#[test]
fn test() {
    let setup: Setup<f32> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Swizzled,
        input_data_format: DataFormat::Complex,
        output_data_format: DataFormat::Complex,
        len: 1024,
        inverse: false,
    })
    .unwrap();
    let setup_rc = Rc::new(setup);
    Env::new(setup_rc);
}
