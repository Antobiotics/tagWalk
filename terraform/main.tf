provider "aws" {
    profile = "fachung"
    region = "${var.aws_region}"
}

data "template_file" "credentials" {
  template = "${file("${path.module}/templates/credentials.tpl")}"

  vars {
    aws_access_key_id = "${var.aws_access_key_id}"
    aws_secret_access_key = "${var.aws_secret_access_key}"
  }
}

resource "aws_security_group" "jupyter_notebook_sg" {
    name = "jupyter_notebook_sg"
    # Open up incoming ssh port
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Open up incoming traffic to port 8888 used by Jupyter Notebook
    ingress {
        from_port   = 8888
        to_port     = 8888
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Open up outbound internet access
    egress {
        from_port   = 0
        to_port     = 0
        protocol    = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }
}


resource "aws_instance" "Fachung" {
    count = 1
    availability_zone = "${var.aws_availability_zone}"
    ami = "ami-a8d2d7ce"
    instance_type = "${var.aws_instance_type}"
    key_name = "fachung.pem"

    tags {
        Name = "Fachung"
    }

    vpc_security_group_ids = ["${aws_security_group.jupyter_notebook_sg.id}"]

    provisioner "file" {
        source      = "configure.sh"
        destination = "/tmp/configure.sh"

        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }
    }

    provisioner "file" {
        content = "${data.template_file.credentials.rendered}"
        destination = "/tmp/credentials"

        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }
    }

    provisioner "remote-exec" {
        inline = [
            "chmod +x /tmp/configure.sh",
            "/tmp/configure.sh",
        ]

        connection {
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }

    }
}


resource "aws_ebs_volume" "FachungData" {
    availability_zone = "eu-west-1b"
    size              = "${var.fachung_volume_size}"

    tags {
        Name = "FachungData"
    }

    /* lifecycle { */
    /*     prevent_destroy = true */
    /* } */
}


resource "aws_volume_attachment" "ebs_att" {
    /* skip_destroy = true */
    device_name = "/dev/sdh"
    volume_id   = "${aws_ebs_volume.FachungData.id}"
    instance_id = "${aws_instance.Fachung.id}"

    provisioner "remote-exec" {
        scripts = [
            "${path.module}/attach-data-volume.sh"
        ]
        connection {
            host     = "${aws_instance.Fachung.public_ip}"
            type     = "ssh"
            user     = "ubuntu"
            private_key = "${file("~/.aws/fachung.pem")}"
        }
    }
}


output "node_dns_name" {
    value = "${aws_instance.Fachung.public_dns}"
}

